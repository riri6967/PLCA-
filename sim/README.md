# Building the OMNeT++ model

After `setup_omnet.sh`, build the model from a folder holding `AdaptivePlca.*`,
`AdaptivePlcaController.cc` and the two headers from `results/omnet/`:

```
opp_makemake -f -s -o plcasim -O out -DINET_IMPORT -I$INET_ROOT/src -L$INET_ROOT/src -lINET
make -j8 MODE=release
```

Copy `libplcasim.so` next to `omnetpp.ini`, then run `python3 run_batch.py <config>`.
