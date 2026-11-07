#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "inet/common/packet/Packet.h"
#include "inet/queueing/contract/IPacketCollection.h"
#include "omnetpp.h"
#include "AdaptivePlca.h"
#include "plca_state_model.h"
#include "plca_state_model_mlp.h"

using namespace omnetpp;

namespace plcasim {

// Same quantities as plca_sim/monitor.py and governor as plca_sim/plan.py.
class AdaptivePlcaController : public cSimpleModule, public cListener
{
  protected:
    enum Mode { OFF, EWMA, TINYML, TREE, WRONG };
    Mode mode = OFF;
    bool useGovernor = true;
    int hysteresis = 2, highBurst = 1, congestedBurst = 3, safetyBytes = 64;
    double safetyPeriodBt = 0, safetyDeadlineBt = 0;
    int numNodes = 0;
    std::vector<int> baseBurst, frameBt;
    std::vector<int> safety;
    std::vector<AdaptivePlca *> plca;
    std::vector<inet::queueing::IPacketCollection *> queues;
    std::vector<int> active;
    cMessage *tick = nullptr;
    simsignal_t cycleSignal, toUsedSignal, sentSignal, rxSignal;
    FILE *log = nullptr;

    // window counters
    double busyBt = 0, lastCycleUs = 0, cycleEwmaUs = 0;
    long tosUsed = 0, tosTotal = 0, arrivals = 0;
    double utilPrev = 0, ratePrev = 0, utilEwma = -1, rateEwma = 0, queueEwma = 0;

    // policy state
    int state = 0, pendingState = 0, pendingCount = 0;
    long proposals = 0, overrides = 0, changes = 0;
    std::vector<int> queued, pending;
    bool hasPending = false;

    static int wire(int bytes) { return (bytes + 8) * 8 + 96; }

    static std::vector<int> parseInts(const char *text)
    {
        std::vector<int> out;
        std::istringstream in(text);
        for (int v; in >> v;)
            out.push_back(v);
        return out;
    }

    double worstCycleBt(const std::vector<int> &burst) const
    {
        double cycle = 20 + 32;  // BEACON and the coordinator's own empty TO
        for (int n = 0; n < numNodes; n++)
            cycle += (1 + burst[n]) * frameBt[n];
        return cycle;
    }

    // Work Plan 4.3, with the queue condition that its formula leaves out.
    bool accepts(const std::vector<int> &burst) const
    {
        double cycle = worstCycleBt(burst);
        for (int n : safety) {
            int room = 1 + burst[n];
            if (std::ceil(cycle / safetyPeriodBt) > room)
                return false;  // the safety queue could grow without limit
            if (cycle + room * wire(safetyBytes) > safetyDeadlineBt)
                return false;
        }
        return true;
    }

    static int stateOf(double util, double queueTotal)
    {
        if (util >= 0.85 || queueTotal >= 10)
            return 3;
        return util >= 0.60 ? 2 : util >= 0.30 ? 1 : 0;
    }

    void apply(const std::vector<int> &burst)
    {
        if (burst == active)
            return;
        changes++;
        active = burst;
        for (int n = 0; n < numNodes; n++)
            plca[n]->setMaxBurst(active[n]);
    }

    void onWindow()
    {
        const double window = par("windowSize").doubleValue();
        const double util = std::min(busyBt / (window * 1e7), 1.0);
        const double toUtil = tosTotal ? double(tosUsed) / tosTotal : 0;
        const double rate = arrivals / (window * 1e3);
        double total = 0, longest = 0, activeNodes = 0, waiting = 0;
        for (int n = 0; n < numNodes; n++) {
            double q = queues[n]->getNumPackets();
            queued[n] = int(q);
            total += q;
            longest = std::max(longest, q);
            activeNodes += q > 0;
        }
        for (int n : safety)
            waiting += queues[n]->getNumPackets();
        const double a = 0.3;
        if (utilEwma < 0) {
            utilEwma = util;
            rateEwma = rate;
            queueEwma = total;
        }
        else {
            utilEwma += a * (util - utilEwma);
            rateEwma += a * (rate - rateEwma);
            queueEwma += a * (total - queueEwma);
        }
        const float f[PLCA_NUM_FEATURES] = {
            float(util), float(utilPrev), float(util - utilPrev), float(toUtil), float(rate),
            float(rate - ratePrev), float(total), float(longest), float(activeNodes), float(waiting),
            float(lastCycleUs), float(cycleEwmaUs), float(utilEwma), float(rateEwma), float(queueEwma)
        };
        if (log) {
            fprintf(log, "%.4f", simTime().dbl());
            for (float v : f)
                fprintf(log, ",%.6g", v);
            fprintf(log, "\n");
        }
        utilPrev = util;
        ratePrev = rate;
        busyBt = 0;
        tosUsed = tosTotal = arrivals = 0;
        if (mode == OFF)
            return;

        int next = mode == TINYML ? plca_predict_mlp(f)
                 : mode == TREE ? plca_predict_state(f)
                 : mode == EWMA ? stateOf(utilEwma, total)
                 : intuniform(0, 3);
        pendingCount = next == pendingState ? pendingCount + 1 : 1;
        pendingState = next;
        if (pendingCount >= hysteresis)
            state = next;

        int extra = state == 3 ? congestedBurst : state == 2 ? highBurst : 0;
        std::vector<int> wanted(baseBurst);
        // Safety bursts stay fixed: lowering one would strand a backlog built up in a long cycle.
        for (int n = 0; n < numNodes; n++)
            if (queued[n] >= 2 && std::find(safety.begin(), safety.end(), n) == safety.end())
                wanted[n] += extra;
        proposals++;
        if (!useGovernor || accepts(wanted)) {
            pending = wanted;
            hasPending = true;
        }
        else
            overrides++;
    }

    void initialize(int stage) override
    {
        if (stage == 0) {
            const char *m = par("mode");
            mode = !strcmp(m, "tinyml") ? TINYML : !strcmp(m, "tree") ? TREE : !strcmp(m, "ewma") ? EWMA : !strcmp(m, "wrong") ? WRONG : OFF;
            useGovernor = par("governor");
            hysteresis = par("hysteresis");
            highBurst = par("highBurst");
            congestedBurst = par("congestedBurst");
            safetyBytes = par("safetyFrameBytes");
            safetyPeriodBt = par("safetyPeriod").doubleValue() * 1e7;
            safetyDeadlineBt = par("safetyDeadline").doubleValue() * 1e7;
            safety = parseInts(par("safetyNodes"));
            numNodes = getParentModule()->par("numNodes");
            baseBurst = parseInts(par("baseBurst"));
            if (baseBurst.empty())
                baseBurst.assign(numNodes, 0);
            frameBt.assign(numNodes, wire(1518));
            std::vector<int> sizes = parseInts(par("maxFrameBytes"));
            for (int n = 0; n < (int)sizes.size() && n < numNodes; n++)
                frameBt[n] = wire(sizes[n]);
            cycleSignal = registerSignal("cycleLength");
            toUsedSignal = registerSignal("transmitOpportunityUsed");
            sentSignal = registerSignal("packetSent");
            rxSignal = registerSignal("packetReceivedFromLower");
            tick = new cMessage("window");
            const char *path = par("windowLog");
            if (*path) {
                log = fopen(path, "a");
                if (!log)
                    throw cRuntimeError("Cannot open %s", path);
            }
        }
        else if (stage == 1) {
            cModule *net = getParentModule();
            for (int n = 0; n < numNodes; n++) {
                cModule *eth = net->getSubmodule("node", n)->getSubmodule("eth", 0);
                auto p = check_and_cast<AdaptivePlca *>(eth->getSubmodule("plca"));
                plca.push_back(p);
                queues.push_back(check_and_cast<inet::queueing::IPacketCollection *>(eth->getSubmodule("queue")));
                p->setMaxBurst(baseBurst[n]);
            }
            active = baseBurst;
            queued.assign(numNodes, 0);
            net->subscribe(cycleSignal, this);
            net->subscribe(toUsedSignal, this);
            net->subscribe(sentSignal, this);
            net->subscribe(rxSignal, this);
            scheduleAt(simTime() + par("windowSize").doubleValue(), tick);
        }
    }

    int numInitStages() const override { return 2; }

    bool atController(cComponent *source) const
    {
        return strstr(source->getFullPath().c_str(), ".controller.") != nullptr;
    }

    void receiveSignal(cComponent *source, simsignal_t id, intval_t value, cObject *) override
    {
        // Each node reports only its own TO, so the coordinator's view alone would read 0.
        if (id == toUsedSignal) {
            tosTotal++;
            tosUsed += value != 0;
        }
    }

    void receiveSignal(cComponent *source, simsignal_t id, const SimTime &value, cObject *) override
    {
        if (id == cycleSignal && atController(source)) {
            lastCycleUs = value.dbl() * 1e6;
            cycleEwmaUs += 0.2 * (lastCycleUs - cycleEwmaUs);
            // Work Plan 4.3: a change mid-cycle could mix two settings into a cycle longer than either.
            if (hasPending) {
                apply(pending);
                hasPending = false;
            }
        }
    }

    void receiveSignal(cComponent *source, simsignal_t id, cObject *obj, cObject *) override
    {
        auto packet = dynamic_cast<inet::Packet *>(obj);
        if (!packet)
            return;
        if (id == sentSignal && !atController(source))
            arrivals++;
        else if (id == rxSignal && atController(source))
            busyBt += packet->getTotalLength().get<inet::b>() + 8 * 8 + 96;
    }

    void handleMessage(cMessage *msg) override
    {
        onWindow();
        scheduleAfter(par("windowSize").doubleValue(), msg);
    }

    void finish() override
    {
        recordScalar("proposals", proposals);
        recordScalar("overrides", overrides);
        recordScalar("changes", changes);
        recordScalar("overrideRate", proposals ? double(overrides) / proposals : 0.0);
        if (log) {
            fclose(log);
            log = nullptr;
        }
    }

  public:
    ~AdaptivePlcaController() override { cancelAndDelete(tick); }
};

Define_Module(AdaptivePlcaController);

} // namespace plcasim
