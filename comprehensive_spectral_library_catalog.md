# Comprehensive public spectral-library catalog for land-cover optical / hyperspectral work

This document compiles public spectral libraries, datasets, repositories, and archives relevant to **vegetation, soils, urban materials, water, snow/ice, and mixed land-cover applications**. A key caveat is that **very few public resources are standardized to exactly 300–2500 nm**; many strong VSWIR libraries begin at **350 nm** or **400 nm**, while broader libraries span well beyond the optical/SWIR window.

## Scope and labeling

- **Spectral type** = the main target/domain the resource is best suited for.
- **Resource type** distinguishes:
  - **direct library / dataset**: primary spectra or labeled spectral records,
  - **derived / harmonized**: processed or repackaged spectra,
  - **portal / archive**: discovery or hosting system rather than one fixed library.
- **Coverage** is listed when it could be verified from the indexed source; otherwise it is described more generally.

## 1) Direct public libraries and datasets

### 1.1 Broad / multi-class reference libraries

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **USGS Spectral Library v7** | Broad reference library: minerals, soils, vegetation, man-made materials, liquids/water, frozen volatiles | **0.2–200 µm** overall; USGS also distributes an ASD-convolved **350–2500 nm** product | Direct library | USGS ([usgs.gov](https://www.usgs.gov/data/usgs-spectral-library-version-7-data)) |
| **ECOSTRESS Spectral Library v1.0** | Broad reference library: vegetation, NPV, rocks, soils, snow/ice, man-made materials | **0.35–15.4 µm** | Direct library | JPL / NASA ([speclib.jpl.nasa.gov](https://speclib.jpl.nasa.gov/)) |
| **ASTER Spectral Library v2.0** | Broad legacy reference library: minerals, rocks, soils, vegetation, snow/ice, man-made materials | **0.4–15.4 µm** | Direct library | JPL / NASA ([speclib.jpl.nasa.gov](https://speclib.jpl.nasa.gov/downloads/2009-Baldridge.pdf)) |

### 1.2 Soil-focused libraries

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **Open Soil Spectral Library (OSSL)** | Global soil library | Vis–NIR typically **350–2500 nm** plus MIR holdings | Direct library | Soil Spectroscopy / Zenodo ([zenodo.org](https://zenodo.org/records/5805138)) |
| **ICRAF Spectral Library** | Soil-dominant library with additional sediments, plants, some man-made materials and rocks | **350–2500 nm** | Direct library | World Agroforestry / ICRAF ([apps.worldagroforestry.org](https://apps.worldagroforestry.org/sensingsoil/Spectral%20Library.asp)) |
| **Brazilian Soil Spectral Library (BSSL)** | National soil library | Vis–NIR–SWIR: **2151 bands / 16,084 rows**; MIR subset also included | Direct library | Zenodo / USP GeoCiS ([zenodo.org](https://zenodo.org/records/8361419)) |
| **LUCAS Soil spectral data** | Pan-European soil spectral dataset | Vis–NIR **400–2500 nm** | Direct dataset | ESDAC / European Commission ([esdac.jrc.ec.europa.eu](https://esdac.jrc.ec.europa.eu/content/predicting-soil-properties-using-spectral-subsets-lucas-visible-near-infrared-spectroscopy)) |
| **Vis–NIR Soil Spectral Library of the Hungarian Soil Degradation Observation System (HSDOS)** | National soil library | **350–2500 nm** | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/11394278)) |
| **ProbeField pre-processed spectra** | Multi-country field soil spectra | **350–2500 nm** | Direct dataset | Zenodo / EJP Soil ([zenodo.org](https://zenodo.org/records/13753159)) |
| **ProbeField aligned spectra** | Cross-instrument aligned soil laboratory spectra | Soil laboratory spectra; aligned using internal soil standard | Derived dataset | Zenodo / EJP Soil ([zenodo.org](https://zenodo.org/records/13757029)) |
| **Near-infrared soil spectral library using the NeoSpectra scanner** | Soil NIR library | Narrower NIR range than classic ASD-style VSWIR | Direct dataset | Zenodo ([zenodo.org](https://zenodo.org/records/7586622)) |

### 1.3 Vegetation, agriculture, and mixed natural-target libraries

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **Spectral library of vegetation from Mediterranean woodlands** | Vegetation / canopy | Vegetation reflectance library from Mediterranean oak woodland | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/5176824)) |
| **Spectral library of plant species from Montesinho Natural Park** | Plant-species / vegetation | Species-level reflectance database | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/10797961)) |
| **Drylands spectral libraries in support of EMIT** | Dryland vegetation / soil / NPV | Selection of contact-probe and leaf-clip spectra from multiple dryland campaigns | Direct library set | EcoSIS ([ecosis.org](https://ecosis.org/package/drylands-spectral-libraries-in-support-of-emit)) |
| **NGEE Arctic 2018 Vegetation Endmember Spectral Reflectance** | Arctic vegetation endmembers | **350–2500 nm** | Direct library | EcoSIS ([ecosis.org](https://ecosis.org/package/ngee-arctic-2018-vegetation-endmember-spectral-reflectance-kougarok-rd-seward-peninsula-alaska)) |
| **UW-BNL NASA HyspIRI California Airborne Campaign Ground Target Spectra** | Mixed natural ground targets (calibration, bare/fallow fields, rock, stable targets) | **350–2500 nm** | Direct library | EcoSIS ([ecosis.org](https://ecosis.org/package/uw-bnl-nasa-hyspiri-california-airborne-campaign-ground-target-spectra)) |
| **Global Hyperspectral Imaging Spectral-library of Agricultural crops for Conterminous United States V001 (GHISACONUS)** | Agricultural crops / vegetation | Hyperion-based crop library for major CONUS crops and growth stages | Direct dataset | NASA Earthdata / LP DAAC ([lpdaac.usgs.gov](https://lpdaac.usgs.gov/documents/609/GHISACONUS_User_Guide_V1.pdf)) |
| **Global Hyperspectral Imaging Spectral-library of Agricultural crops for Central Asia V001 (GHISACASIA)** | Agricultural crops / vegetation | Hyperion + field ASD crop library for Central Asia irrigated agriculture | Direct dataset | NASA Earthdata / LP DAAC ([earthdata.nasa.gov](https://www.earthdata.nasa.gov/data/catalog/lpcloud-ghisacasia-001)) |

### 1.4 Urban / man-made material libraries

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **Spectral Library of Impervious Urban Materials (SLUM)** | Urban impervious materials | **300–2500 nm** reflectance + **8–14 µm** emissivity | Direct library | Zenodo / LUMA ([zenodo.org](https://zenodo.org/records/4263842)) |
| **Karlsruhe Library of Urban Materials (KLUM)** | Urban VNIR–SWIR materials, especially facades plus roofs and ground materials | VNIR–SWIR | Direct library | Zenodo + data paper ([zenodo.org](https://zenodo.org/records/3441838)) |
| **Walloon Roof Material (WaRM) spectral library** | Roof materials | **350–2500 nm** | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/7414740)) |
| **Spectral Library of Rooftop Urban Materials (SLyRUM)** | Rooftop urban materials | Mediterranean-region rooftop materials | Direct library | UAB / UAB repository ([ddd.uab.cat](https://ddd.uab.cat/pub/dadrec/2017/196065/SLyRUM_Zambrano_Josa_Rieradevall_Gasso_Gabarrell.pdf)) |
| **Urban Reflectance Spectra from Santa Barbara, CA** | Mixed urban surfaces + soil + GV + NPV | ASD field spectra from Santa Barbara | Direct library | EcoSIS ([ecosis.org](https://ecosis.org/package/urban-reflectance-spectra-from-santa-barbara--ca)) |
| **Santa Barbara asphalt road spectra library** | Asphalt roads / pavement condition | Road-surface spectral library | Direct library | Referenced literature / Santa Barbara study context ([ugpti.org](https://www.ugpti.org/smartse/research/citations/downloads/Herold-Understanding_Spectral_Characteristics_Asphalt_Roads-2004.pdf)) |
| **Urban Materials Spectral Library v1.0** | Small urban-material library (asphalt, brick, pavers) | Around **350–1000 nm** classically, collected with handheld spectrometer | Direct library | Wright State CORE Scholar ([corescholar.libraries.wright.edu](https://corescholar.libraries.wright.edu/spectral_data/1/)) |
| **urbisphere urban hyperspectral library: Berlin** | Urban surface materials | Urban spectral reflectance library | Direct library | Zenodo / urbisphere project ([zenodo.org](https://zenodo.org/records/8032782/files/119.pdf?download=1)) |
| **German image spectral library of urban surface materials** | Urban image spectra | **455–2449 nm** | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/12192460)) |
| **Brussels image-based library of urban materials** | Urban image spectra | VNIR–SWIR library of **1274** urban cover spectra from APEX Brussels 2015 | Direct library | Zenodo ([zenodo.org](https://zenodo.org/records/6472398)) |
| **Urban material ground truth data for the 2015 APEX Brussels image** | Urban image spectra / labeled ground truth | **450–2431 nm**, **1350** labeled spectra | Direct dataset | Zenodo ([zenodo.org](https://zenodo.org/records/11181008)) |
| **Urban material ground truth data for the 2007 HyMap image** | Urban image spectra / labeled ground truth | Image-derived reflectance spectra of typical urban materials | Direct dataset | Zenodo ([zenodo.org](https://zenodo.org/records/11193356)) |

### 1.5 Water / aquatic resources

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **SeaSWIR** | Turbid/coastal water reflectance | Marine reflectance focused on SWIR-I / SWIR-II; project covers **1000–2500 nm** behavior, with 137 ASD spectra | Direct dataset | ESSD / PANGAEA ([essd.copernicus.org](https://essd.copernicus.org/articles/10/1439/2018/)) |
| **GLORIA** | Aquatic hyperspectral reflectance + water quality | **350–900 nm**, 1 nm intervals | Direct dataset | Vrije Universiteit Amsterdam / PANGAEA ([research.vu.nl](https://research.vu.nl/en/datasets/gloria-a-global-dataset-of-remote-sensing-reflectance-and-water-q/)) |
| **PANTHYR O1BE hyperspectral water reflectance** | Autonomous water reflectance measurements | Water-leaving radiance reflectance from autonomous PANTHYR station | Direct dataset | Zenodo / WaterHypernet context ([zenodo.org](https://zenodo.org/records/10024143)) |

### 1.6 Snow / ice resources

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **SISpec (Snow and Ice Spectral Library)** | Snow / ice | Snow and ice spectral library for hyperspectral measurements | Direct library | SISpec / Zenodo metadata ecosystem ([zenodo.org](https://zenodo.org/records/4812454/files/SISPEC-NetCDF%20encoding%20v1.0.pdf)) |
| **Laboratory spectral reflectance measurements of natural snow samples and pine/spruce twigs** | Snow-focused reflectance with vegetation context | **350–2500 nm** | Direct dataset | Zenodo ([zenodo.org](https://zenodo.org/records/2677477)) |

## 2) Derived, harmonized, or repackaged resources

These are very useful, but they are **not the same thing as raw primary spectral libraries**.

| Library / name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **EMIT L2A surface repository (`emit-sds-l2a/surface`)** | EMIT surface-model priors for atmospheric correction | EMIT-oriented VSWIR surface priors; repo notes spectra come mostly from USGS v7 with some additions and manual adjustments | **Derived / curated repository** | GitHub + linked EcoSIS packages ([github.com](https://github.com/emit-sds/emit-sds-l2a)) |
| **EMIT manually adjusted vegetation reflectance spectra** | Vegetation priors for EMIT | Adjusted from public source spectra; designed to be statistically representative rather than diagnostic originals | Derived library | EcoSIS ([ecosis.org](https://ecosis.org/package/emit-manually-adjusted-vegetation-reflectance-spectra)) |
| **EMIT manually adjusted snow and liquids reflectance spectra** | Snow / liquids priors for EMIT | Adjusted from USGS v7 and related source spectra | Derived library | EcoSIS ([ecosis.org](https://ecosis.org/package/emit-manually-adjusted-snow-and-liquids-reflectance-spectra)) |
| **EMIT manually adjusted surface reflectance spectra** | Mixed surface priors for EMIT | Heavily altered from original measurements for atmospheric-correction statistics | Derived library | EcoSIS ([ecosis.org](https://ecosis.org/package/emit-manually-adjusted-surface-reflectance-spectra)) |
| **earthlib** | Harmonized land-cover endmember library | **400–2450 nm** at **10 nm** band widths; vegetation, soil, NPV, urban, burned materials | Harmonized library | earthlib docs / repo / Zenodo release ([earth-chris.github.io](https://earth-chris.github.io/earthlib/)) |
| **Existing Urban Hyperspectral Reference Data** | FAIR repackage of prior urban libraries | Converted **9 pre-existing** urban libraries into organized `.xlsx` packages | Repackaged compilation | Zenodo ([zenodo.org](https://zenodo.org/records/10963442)) |

## 3) Portals, archives, and discovery systems

These are important for a comprehensive workflow, but they are **platforms or archives**, not single fixed spectral libraries.

| Name | Spectral type | Coverage | Resource type | Source |
|---|---|---:|---|---|
| **EcoSIS** | Multi-domain ecological spectra | Large repository of hosted spectral packages | Portal / archive | EcoSIS ([ecosis.org](https://ecosis.org/)) |
| **SPECCHIO** | General spectral data and campaign management | Reference spectra and campaign data with rich metadata | Portal / information system | SPECCHIO ([specchio.ch](https://specchio.ch/)) |
| **Australian National Spectral Database (NSD)** | National multi-target spectral repository | Targets include soils, vegetation, waterbodies, coral, algae, seagrasses, and land surfaces | Portal / curated national database | Geoscience Australia / DEA ([ga.gov.au](https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/national-spectral-database)) |
| **SeaBASS** | In situ oceanographic / aquatic optics archive | Public archive of ocean and atmosphere in situ data maintained by NASA OBPG | Archive | NASA / Earthdata ([seabass.gsfc.nasa.gov](https://seabass.gsfc.nasa.gov/)) |
| **WaterHypernet** | Automated hyperspectral water reflectance network | QC-approved hyperspectral water reflectance products across network sites | Archive / network | WaterHypernet ([waterhypernet.org](https://waterhypernet.org/data/)) |

## 4) Known but not currently public / not ready for download

| Name | Spectral type | Status | Source |
|---|---|---|---|
| **Berlin Urban Materials Spectral Library** | Urban materials | Record exists, but indexed record shows **0 bytes / no downloadable dataset** in current public view, so it is not counted here as an available library yet | Zenodo ([zenodo.org](https://zenodo.org/records/11385631)) |

## 5) Recommended “master stack” for this use case

For a **comprehensive land-cover spectral reference set** focused on **vegetation, soil, urban, water, snow/ice**, the following is a practical backbone:

- **USGS v7** for the broadest cross-class base.
- **ECOSTRESS / ASTER** for broad public coverage and strong vegetation / snow / man-made support.
- **OSSL, ICRAF, BSSL, LUCAS, HSDOS, ProbeField** for soils.
- **SLUM, KLUM, WaRM, SLyRUM, Santa Barbara, Brussels/APEX/HyMap urban libraries** for urban materials.
- **SeaSWIR, GLORIA, PANTHYR, SeaBASS / WaterHypernet** for water and aquatic reflectance.
- **SISpec** plus the **natural snow samples** dataset for snow/ice.
- **GHISACONUS / GHISACASIA** for crop-focused agricultural spectra.
- **EMIT surface** and **earthlib** as **derived / harmonized layers** on top of the primary libraries, not replacements for them.

## 6) Important caveats

1. **Exact 300–2500 nm coverage is uncommon.**  
   The clearest exact public urban match in this catalog is **SLUM**; many of the strongest vegetation and soil libraries begin at **350 nm**, and several major legacy libraries begin at **400 nm**.

2. **Not every item here is a raw field/lab library.**  
   Some are **image-derived spectral libraries** (for example, the Brussels/APEX/HyMap urban sets), some are **archives** (SeaBASS, WaterHypernet), and some are **derived / adjusted** (EMIT surface, earthlib).

3. **This is comprehensive, but not literally every niche spectral dataset worldwide.**  
   The focus here is on **public, verifiable resources** that are useful for the target domains and that could be confirmed from official project pages, repository records, or primary documentation.
