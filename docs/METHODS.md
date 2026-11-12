# Methods

The analysis simulates the addition of $P = 3$ new Kitas in Münster and compares where an efficiency objective and two
Rawlsian-inspired objectives would place them. All three models use the same inputs.

## 1. Data

| Data | Source | Reference date | Role |
| --- | --- | --- | --- |
| Projected children aged 0-2 and 3-5 per district | Stadt Münster, small-area forecast | 2026 | district totals of demand |
| Children aged under 3 and 3-5 per 100 m cell | Census 2022 grid (Destatis) | 15 May 2022 | where children live within a district |
| Residential and mixed-use buildings | ALKIS cadastre (Geobasis NRW) | 2023-2026 | residential locations and footprints |
| Four social indicators per district | Stadt Münster | 31 Dec 2025 | vulnerability index |
| Walking network and existing Kitas (`amenity=kindergarten`) | OpenStreetMap | 20 Jun 2026 | walking times and current supply |

Study area: the 45 statistical districts of Münster. Of 57,006 residential buildings, 41 documented student-only
residences are excluded (no young children), leaving 56,965 eligible buildings.

## 2. Estimated childcare demand

**What and why.** Demand is the estimated number of children aged 0-5 in 2026 at each residential location. The
forecast exists only for districts, but walking times differ within districts, so the district totals are distributed
using the only observed small-area distribution of young children, the Census 2022 grid. This is **potential
residential demand**, not enrolment or demand for places.

**How.** For district $d$ with projected children $C_d$, census cell $g$ (assigned to the district containing its centre)
with census children $N_g$, and eligible building $b$ in that cell with footprint $A_b$:

$$
D_g = C_d\,\frac{N_g}{\sum_{h\in d}N_h},\qquad
D_b = D_g\,\frac{A_b}{\sum_{k\in g}A_k},\qquad
D_i=\sum_{b\in i}D_b .
$$

Census values suppressed for confidentiality ("–") are read as 0. A cell with census children but no eligible building
keeps its children at the cell centre (43 cells, 320 children). Buildings and cell centres are attached to their
nearest network node; a **demand location** $i$ is a node within one district, with demand $D_i$. The allocation
conserves every district total exactly.

**Result.** 14,960 children at 11,275 demand locations; 55.7% of locations carry fewer than one estimated child
(median 0.87). One cell contains the state reception centre for asylum seekers (ZUE) in Gremmendorf-West; its
107.4 projected children are verified census population and are included.

**Why this allocation.** Distributing district totals over all buildings by footprint would place 41.6% of the children
in cells where the census reports none. Within a 100 m cell, footprint is a plausible divider.

## 3. Pedestrian network and access

Walking times are computed on a pedestrian network built from the OpenStreetMap snapshot of 20 June 2026 (62,284 nodes),
at a constant 5 km/h. Access rules follow OSM tags (`foot` before `access` before the default of the road type):

- **Excluded:** motorroads (*Kraftfahrstraßen*, where walking is prohibited), ways tagged `access=no` or `foot=no`, and
  cycle-only paths. The standard OSMnx walking network includes motorroads, which would create an artificial worst Q4
  location.
- **Destination-only:** private and other restricted ways and gates, usable only at the start or end of a trip.
- **Sidewalk proxy (assumption):** roads whose pedestrians must use a separately mapped sidewalk are kept as a proxy for
  that sidewalk, because OSM maps crossings and sidewalk continuity incompletely. This may slightly underestimate walks
  where a road cannot be crossed.

A location's **access** $t_i$ is its shortest walking time to the nearest open Kita. Current access ($c_i$) uses the
227 existing Kitas; capacities are not modelled.

## 4. Vulnerability and the priority group (Q4)

The childcare-vulnerability index of district $d$ is the equal-weight mean of four min-max-normalised 2025 indicators
$x_{dk}$: the shares of SGB II recipients aged 15-64 and aged 0-14, of children with a migration history (a proxy for
access barriers), and of children in single-parent households:

$$
v_d=\frac14\sum_{k=1}^{4}\frac{x_{dk}-\min_{d'}x_{d'k}}{\max_{d'}x_{d'k}-\min_{d'}x_{d'k}} .
$$

**Q4** is the set of districts with $v_d$ at or above the 75th percentile: 12 districts with 3,490 locations and 5,323
estimated children (35.6%). Every location in a Q4 district is a Q4 location. The twelfth district (Gievenbeck) is only
0.001 above the thirteenth (Uppenberg).

## 5. Candidate sites

Every network node of an eligible residential building is a candidate site: **18,278 candidates**. They represent
possible locations, not verified sites. For each location only candidates that would shorten its current walk are kept
($d_{ij} < c_i$).

## 6. The three models

Notation: $S$ is the set of $P$ opened candidates; $t_i(S)=\min\{c_i, \min_{j\in S} d_{ij}\}$ is the walking time to the
nearest open Kita; $\mathcal{I}_Q$ the Q4 locations; $\mathcal{Q}$ the Q4 districts.

| Model | Objective | Role of estimated demand $D_i$ |
| --- | --- | --- |
| Utilitarian | $\min_S \sum_i D_i\, t_i(S)$ (total child-minutes) | weight of every location |
| Individual Rawlsian | Stage 1: $Z^*=\min_S \max_{i\in\mathcal{I}_Q} t_i(S)$; Stage 2: $\min_S \sum_{i\in\mathcal{I}_Q}(t_i(S)-\bar t)^2$ subject to $\max_{i\in\mathcal{I}_Q} t_i(S) \le Z^*$ (up to a rounding tolerance) | **none**: demand only identifies the eligible locations ($D_i>0$) |
| District Rawlsian | $\min_S \max_{d\in\mathcal{Q}} A_d(S)$, with $A_d(S)=\sum_{i\in d}D_i\,t_i(S)/\sum_{i\in d}D_i$ | weight within each Q4 district mean |

- **Utilitarian** (p-median type): every estimated child counts equally; no priority for Q4.
- **Individual Rawlsian** (p-center type): an **unweighted, location-based maximin**. Every Q4 location counts the same,
  whatever its number of children; Stage 2 then reduces the spread of Q4 walking times without worsening the worst one.
- **District Rawlsian**: protects the Q4 district whose children walk longest on average. It protects district
  averages, so it does not bound individual walks.

The models are "Rawlsian-inspired": they adopt only the maximin structure of Rawls's difference principle, with walking
time as the advantage and Q4 as the least advantaged group.

**Solution.** All models are mixed-integer programmes solved with Gurobi 13.0.2.
