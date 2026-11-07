#ifndef PLCA_ADAPTIVEPLCA_H
#define PLCA_ADAPTIVEPLCA_H

#include "inet/physicallayer/wired/ethernet/EthernetPlca.h"

namespace plcasim {

// INET's PLCA reads max_bc once at start. It re-reads the field at every burst
// decision, so setting it at the BEACON behaves like a management write to a
// real PHY's MaxBurstCount register.
class AdaptivePlca : public inet::physicallayer::EthernetPlca
{
  public:
    void setMaxBurst(int value) { max_bc = value; }
    int getMaxBurst() const { return max_bc; }
};

} // namespace plcasim

#endif
