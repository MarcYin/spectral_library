# Vegetation Spectral Library Candidates

This note collects vegetation-heavy libraries that can extend the current SIAC
build, especially for crops, trees, leaves, tundra plants, and forest
understory spectra.

The current native-range-filtered SIAC package
`build/siac_spectral_library_v14_native_410_2490_no_ghisacasia` retains only
`2585` vegetation spectra, so the main goal here is to add more plant-dominant
sources rather than more broad mixed libraries.

## Added To Manifest

These sources were added to `manifests/sources.csv`.

### Immediate high-value candidate

- `cabo_leaf_v2`
  - Title: [CABO 2018-2019 Leaf-Level Spectra v2](https://ecosis.org/package/a12286ac-929f-4e3e-9b9c-517e10fe4087)
  - Why it matters: leaf-level plant spectra spanning crops, trees, grasses, and forbs
  - Access: public EcoSIS package UUID
  - Fetch status: suitable for the existing `ecosis_package` fetcher
  - Notes: package exposes separate `ref_spec.csv`, `trans_spec.csv`, and `abs_spec.csv` resources

### Public sources that still need fetch/path resolution

- `neon_field_spectra`
  - Title: [NEON Field Spectra (DP1.30012.001)](https://data.neonscience.org/data-products/DP1.30012.001)
  - Focus: foliar / field spectrometer vegetation data from NEON sites
  - Why it matters: broad species and site coverage, strong tree and foliage value
  - Notes: public product page, but ingest should be done through a dedicated NEON product/API fetch path

- `antarctic_vegetation_speclib`
  - Title: [Spectral Library of Antarctic Terrestrial Vegetation Species](https://datashare.ed.ac.uk/handle/10283/8763)
  - Focus: mosses, lichens, vascular plants, green algae, associated rocks
  - Why it matters: unusual vegetation classes with public 350-2500 nm spectra

- `ngee_arctic_leaf_reflectance_barrow_2013`
  - Title: [NGEE Arctic Leaf Spectral Reflectance Utqiagvik (Barrow) Alaska 2013](https://www.osti.gov/biblio/1441203)
  - Focus: Arctic plant leaf reflectance
  - Why it matters: vegetation-rich tundra leaf library at 350-2500 nm

- `ngee_arctic_leaf_reflectance_kougarok_2016`
  - Title: [NGEE Arctic Leaf Spectral Reflectance Kougarok Alaska 2016](https://www.osti.gov/biblio/1430079)
  - Focus: Arctic shrub and tundra species leaf reflectance
  - Why it matters: complements existing NGEE vegetation endmembers with raw leaf spectra

- `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016`
  - Title: [NGEE Arctic Leaf Spectral Reflectance and Transmittance Barrow Alaska 2014-2016](https://www.osti.gov/biblio/1437044)
  - Focus: reflectance and transmittance for dominant tundra species
  - Why it matters: dense leaf-level Arctic plant dataset with direct reflectance value

- `branch_tree_spectra_boreal_temperate`
  - Title: [A dataset of branch reflectance spectra for seven boreal and temperate tree species](https://data.mendeley.com/datasets/kvnx6vt8x9/1)
  - Focus: branch-level tree spectra across boreal and temperate species
  - Why it matters: adds woody/tree structure not covered by leaf-only libraries

- `understory_estonia_czech`
  - Title: [Dataset of tree canopy structure, understory reflectance spectra and fractional cover in hemiboreal and temperate forest areas in Estonia and the Czech Republic](https://data.mendeley.com/datasets/9dx32rszp8)
  - Focus: forest understory reflectance
  - Why it matters: adds low-stature vegetation and mixed forest-floor conditions

- `understory_icos_europe`
  - Title: [Dataset of understory reflectance measurements across 40 ICOS sites and additional sites in Europe](https://data.mendeley.com/datasets/m97y3kbvt8/1)
  - Focus: understory vegetation across many forest ecosystems
  - Why it matters: strong geographic and ecological diversity for vegetation prototypes

- `shift_dried_ground_leaf`
  - Title: [SHIFT: Reflectance Measurements for Dried and Ground Leaf Materials](https://daac.ornl.gov/SHIFT/guides/SHIFT_DriedGround_Leaf_Reflec.html)
  - Focus: dried / ground leaf material spectra
  - Why it matters: useful for chemistry-informed vegetation variability, though not intact-leaf structure

## Additional Vegetation Leads

These showed up as strong vegetation package titles in current web search, but
their old EcoSIS title-slug URLs currently resolve to invalid package pages in
local checks. They are worth keeping as leads, but should not be added as
fetchable manifest records until their stable package IDs are resolved.

- 2018 Talladega National Forest leaf-level reflectance spectra and foliar traits
- Leaf spectral reflectance and maximum carboxylation rates for tree and crop species collected in Madison, Wisconsin
- Hyperspectral leaf reflectance, biochemistry, and physiology of droughted and watered crops
- Leaf spectra of plants in the Cedar Creek Biodiversity Experiment
- Leaf spectra of old fields at Cedar Creek LTER
- Leaf spectra from forest and biodiversity experiment at Cedar Creek LTER
- Leaf spectra of potted tree species from two boreal and temperate forests in Alberta

## Recommended Ingest Order

1. `cabo_leaf_v2`
2. `neon_field_spectra`
3. `ngee_arctic_leaf_reflectance_kougarok_2016`
4. `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016`
5. `antarctic_vegetation_speclib`
6. `understory_estonia_czech`
7. `branch_tree_spectra_boreal_temperate`

## Notes

- `cabo_leaf_v2` is the cleanest next target because it already has a working
  public EcoSIS package UUID.
- The NGEE Arctic leaf datasets are strong additions because they are public,
  vegetation-dominant, and explicitly 350-2500 nm.
- The two Mendeley understory/tree datasets are useful for filling forest
  vegetation modes that are currently underrepresented in the SIAC package.
