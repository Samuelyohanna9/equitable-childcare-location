# Münster results and limitations

## 1. Results (P = 3)

All walking times are in minutes. The indicators describe the selected sites of each model.

| Indicator | Current | Utilitarian | Individual Rawlsian | District Rawlsian |
| --- | ---: | ---: | ---: | ---: |
| City demand-weighted mean | 5.527 | **5.285** | 5.422 | 5.354 |
| Q4 demand-weighted mean | 5.417 | 5.020 | 5.191 | **4.930** |
| Q4 SD of walking times | 3.822 | 3.366 | **3.219** | 3.330 |
| Worst Q4 location | 24.532 | 21.786 | **18.929** | 21.786 |
| Worst Q4 district mean | 8.478 | 6.058 | 8.202 | **5.716** |
| City child-minutes saved | – | **3,624.5** | 1,575.3 | 2,589.0 |
| Share of savings in Q4 | – | 58% | 76% | **100%** |
| Children with a shorter walk | – | 500 | 181 | **577** |

| Model | Selected sites (district) |
| --- | --- |
| Utilitarian | Mecklenbeck, Berg Fidel, Gremmendorf-West |
| Individual Rawlsian | Gievenbeck, Hiltrup-Mitte, Gremmendorf-West |
| District Rawlsian | Angelmodde, Berg Fidel, Gremmendorf-West (two sites shared with the Utilitarian model) |

**Interpretation.**

- **Efficiency.** The Utilitarian sites save the most child-minutes; 58% of the savings fall in Q4 although it gives Q4
  no priority.
- **Worst-served district.** The District Rawlsian model lowers the worst Q4 district mean most (8.478 to 5.716) and
  places all savings in Q4. Compared with the Utilitarian sites it gives up 2.17 city child-minutes per additional Q4
  child-minute saved, because one site moves from Mecklenbeck (outside Q4) to Angelmodde (Q4).
- **Worst-served location.** The Individual Rawlsian model gives the shortest worst Q4 walk (18.929) but improves the
  walk of only 181 children, because its objective ignores how many children live at a location. It is the costliest
  option for both the city and Q4.
- **Conclusion.** The two Rawlsian readings lead to different sites and different beneficiaries. Bringing estimated
  demand into the equity objective (District model) protects the worst-served vulnerable district at a moderate
  citywide cost.

## 2. Limitations

**Demand.**

- Demand is projected (2026) and distributed by the 2022 census pattern; housing built since 2022 receives no demand.
- It is potential residential demand for ages 0-5, not enrolment, take-up or demand for places.
- Most locations carry fractions of a child; these are expected values, not head-counts.
- The reception-centre cell (107.4 children) is legitimate census population, but its relevance for 2026 is uncertain.

**Vulnerability.**

- Q4 is a classification of whole districts: it includes non-disadvantaged households and excludes vulnerable households
  elsewhere. The twelfth Q4 district is decided by a margin of 0.001.
- Vulnerability and poor access overlap only weakly (rank correlation 0.21). The longest walks in the city lie outside
  Q4, so the Rawlsian objectives cannot improve them.

**Accessibility.**

- Walking only, at a constant 5 km/h, on OSM data, which may miss paths, crossings or barriers. Sidewalk-proxy roads
  may slightly underestimate walks.
- Kitas have no capacities: every child is assigned to the nearest open Kita, so shortages are invisible. Results
  describe potential spatial accessibility, not the supply of places.
- The OSM Kita data contain some likely duplicates and childminders.
- Candidate sites are residential locations used as a proxy; land availability, zoning and costs are not considered.

**Models.**

- The District Rawlsian model protects district averages, not individuals, and depends on the within-district demand
  allocation.
- The models apply a maximin rule to walking time for an area-defined group.
