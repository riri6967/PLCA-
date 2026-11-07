# Phase 0: sources checked

| Source | What it is | Adaptive | Predictive | Guarantee | Standard PHY |
|---|---|---|---|---|---|
| EP 4 401 363 A1 | Filed 22 Mar 2023. Applicant **IC Plus Corp.**, not Realtek as the Work Plan says. A node in emergency mode sends an interference packet to end a lower-priority node's TO early. | Event-driven | No | None given | No: nodes inject interference |
| onsemi Precedence Mode | Vendor feature on NCV7410 and NCN26010. Lower PLCA IDs take priority over higher ones. One distributor calls it non-standard. | No | No | Not stated | Vendor extension |
| Ciraolo 2024, PoliTo MSc thesis | "Automotive Ethernet: Construction and Performance Analysis of a 10BASE-T1S Network". Compares PLCA and CSMA/CD on a test network with Vehicle Spy 3. Only the record and front matter were read. | No | No | No | Yes |

Links:
- https://data.epo.org/publication-server/rest/v1.2/patents/EP4401363NWA1/document.html
- https://www.onsemi.com/products/interfaces/ethernet-controllers/ncv7410
- https://webthesis.biblio.polito.it/30927
