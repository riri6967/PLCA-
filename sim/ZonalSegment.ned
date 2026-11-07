import inet.node.ethernet.EthernetMultidropLink;
import inet.node.ethernet.EthernetPlcaHost;
import inet.physicallayer.wired.common.WireJunction;

//
// One zonal 10BASE-T1S segment: the zone controller (PLCA node 0) and
// numNodes follower ECUs on a shared multidrop cable. Each follower is one
// drop off a backbone of 2 m sections, so 8 followers stay inside the 25 m reach.
//
network ZonalSegment
{
    parameters:
        int numNodes = default(8);
        double backboneLength @unit(m) = default(2m);
        double dropLength @unit(m) = default(0.5m);
        bool adaptive = default(false);
    submodules:
        controller: EthernetPlcaHost {
            parameters:
                eth[*].plca.plca_node_count = default(parent.numNodes + 1);
                eth[*].plca.local_nodeID = default(0);
        }
        node[numNodes]: EthernetPlcaHost {
            parameters:
                eth[*].plca.plca_node_count = default(parent.numNodes + 1);
                eth[*].plca.local_nodeID = default(this.index + 1);
        }
        j[numNodes]: WireJunction;
        controllerLogic: AdaptivePlcaController if adaptive;
    connections:
        controller.ethg++ <--> EthernetMultidropLink { length = parent.dropLength; } <--> j[0].port++;
        for i=0..numNodes-2 {
            j[i].port++ <--> EthernetMultidropLink { length = parent.backboneLength; } <--> j[i+1].port++;
        }
        for i=0..numNodes-1 {
            j[i].port++ <--> EthernetMultidropLink { length = parent.dropLength; } <--> node[i].ethg++;
        }
}
