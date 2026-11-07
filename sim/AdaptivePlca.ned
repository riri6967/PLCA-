import inet.physicallayer.wired.ethernet.EthernetPlca;

//
// EthernetPlca whose burst limit can change during a run.
//
simple AdaptivePlca extends EthernetPlca
{
    parameters:
        @class(plcasim::AdaptivePlca);
}

//
// Work Plan 4.4. Every window: read stats, build features, predict the state,
// apply the rule table, and let the governor accept or reject the result.
//
simple AdaptivePlcaController
{
    parameters:
        @class(plcasim::AdaptivePlcaController);
        // "off" keeps baseBurst, "ewma" uses the measured state, "tinyml" the
        // int8 MLP, "tree" the decision tree, "wrong" a random state (to show the governor holds the bound).
        string mode = default("off");
        double windowSize @unit(s) = default(5ms);
        bool governor = default(true);
        int hysteresis = default(2);
        int highBurst = default(1);
        int congestedBurst = default(3);
        // Base burst count per follower, space separated.
        string baseBurst = default("");
        // Largest frame of each follower in bytes, space separated.
        string maxFrameBytes = default("");
        // Safety followers, space separated, with their common period and deadline.
        string safetyNodes = default("");
        double safetyPeriod @unit(s) = default(1ms);
        double safetyDeadline @unit(s) = default(10ms);
        int safetyFrameBytes = default(64);
        // CSV of one feature row per window, for the TinyML dataset. Empty = off.
        string windowLog = default("");
}
