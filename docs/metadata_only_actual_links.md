# Metadata-Only Sources: Resolved Actual Links

Snapshot from the current local catalog build on 2026-03-08. Updated after checking the USGS data release and attached-file listing.

## Direct asset links found

| source_id | name | actual link(s) | notes |
|---|---|---|---|
| `bssl` | Brazilian Soil Spectral Library (BSSL) | `https://zenodo.org/api/records/8361419/files/BSSL_DB_V002.xlsx/content` | Primary spectral database workbook. |
| `drylands_emit` | Drylands spectral libraries in support of EMIT | `https://data.ecosis.org/dataset/a360734b-df9c-4c55-b2b1-b84e04def064/resource/fd83a444-7a09-4a7e-ab77-c887b9b2563a/download/simulation_asd_data.csv` and `https://data.ecosis.org/dataset/a360734b-df9c-4c55-b2b1-b84e04def064/resource/817b0c57-388b-452b-b3c5-acea8d04203c/download/simulation_asd_metadata.csv` | Resolved through the EcoSIS CKAN API at `data.ecosis.org`. |
| `hsdos` | HSDOS soil library | `https://zenodo.org/api/records/11394278/files/HSDOS_SSL_ver1.0.csv/content` | Main soil spectral CSV. |
| `hyspiri_ground_targets` | UW-BNL NASA HyspIRI California Airborne Campaign Ground Target Spectra | `https://data.ecosis.org/dataset/3c43dcb1-feb3-4a8b-8d34-fb493c2dba2e/resource/915620ea-6073-45a3-a6f0-aa3e32f59831/download/ivanpah_dry_lake_spectra_20130603.zip` | One resolved spectral ZIP from the EcoSIS package. |
| `klum` | Karlsruhe Library of Urban Materials (KLUM) | `https://zenodo.org/api/records/3441838/files/rebeccailehag/KLUM_library-v1.0.zip/content` | Main ZIP bundle. |
| `mediterranean_woodlands` | Spectral library of vegetation from Mediterranean woodlands | `https://zenodo.org/api/records/5176824/files/specveg_data_spectra.txt/content` and `https://zenodo.org/api/records/5176824/files/specveg_metadata.txt/content` | Main spectra plus metadata text tables. |
| `montesinho_plants` | Spectral library of plant species from Montesinho Natural Park | `https://zenodo.org/api/records/10798148/files/Spectrum.zip/content` | The original record `10797961` is restricted and exposes no public files. The open version `10798148` contains `Spectrum.zip`, plus supporting `Spectral_Library.accdb` and `.docx` metadata files. |
| `natural_snow_twigs` | Natural snow samples and pine/spruce twigs | `https://zenodo.org/api/records/2677477/files/Darklab_snow_reflectance_avg.csv/content`, `https://zenodo.org/api/records/2677477/files/Darklab_snow_samplewise_reflectance_avg.csv/content`, `https://zenodo.org/api/records/2677477/files/Darklab_pine_and_spruce_reflectance_all.csv/content` | Several spectral CSVs are available; photos are separate. |
| `neospectra_soil_nir` | NeoSpectra soil NIR library | `https://zenodo.org/api/records/7586622/files/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv/content` and `https://zenodo.org/api/records/7586622/files/Neospectra_WoodwellKSSL_reps_soil+site+NIR.csv/content` | Main NIR spectral tables. |
| `ngee_arctic_2018` | NGEE Arctic 2018 Vegetation Endmember Spectral Reflectance | `https://data.ecosis.org/dataset/4be75211-17df-4807-be3a-7f2a0db166a7/resource/fe6f9265-601a-489d-9c96-7cb3ba921077/download/sewpen_2018_vegendmember_spectra.csv` and `https://data.ecosis.org/dataset/4be75211-17df-4807-be3a-7f2a0db166a7/resource/1d8df61a-0d70-4535-9d74-66fefa7d19c6/download/sewpen_2018_vegendmember_spectra_metadata.csv` | Resolved through the EcoSIS CKAN API. |
| `ossl` | Open Soil Spectral Library (OSSL) | `https://zenodo.org/api/records/5805138/files/ossl_visnir_v1.rds/content`, `https://zenodo.org/api/records/5805138/files/ossl_soillab_v1.rds/content`, `https://zenodo.org/api/records/5805138/files/ossl_soilsite_v1.rds/content` | Main core data objects. `ossl_models.zip` is also exposed but very large. |
| `panthyr_o1be` | PANTHYR O1BE hyperspectral water reflectance | `https://zenodo.org/api/records/10024143/files/PANTHYR-O1BE-v1.0.0.zip/content` | Main ZIP bundle. |
| `probefield_aligned` | ProbeField aligned spectra | `https://zenodo.org/api/records/13757029/files/D3.3_20230919_ProbeField_Aligned_Spectra_V1.csv/content` | Main aligned spectral CSV. |
| `probefield_preprocessed` | ProbeField pre-processed spectra | `https://zenodo.org/api/records/13753159/files/D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_raw_V1.csv/content`, `https://zenodo.org/api/records/13753159/files/D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_EPO_V1.csv/content`, `https://zenodo.org/api/records/13753159/files/D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_ISS_V1.csv/content` | Main spectral CSV tables. |
| `santa_barbara_urban_reflectance` | Urban Reflectance Spectra from Santa Barbara, CA | `https://data.ecosis.org/dataset/94527a02-b72d-4644-aa8b-45074489b7ec/resource/25463d25-37e8-4675-b339-8c05048d5561/download/urbanspectraandmeta.csv` | Direct CSV from the EcoSIS CKAN API. |

## Actual page links found, but not a clean direct asset yet

| source_id | name | actual link | notes |
|---|---|---|---|
| `ecostress_v1` | ECOSTRESS Spectral Library v1.0 | `https://speclib.jpl.nasa.gov/download` | This is the actual JPL download page, but it uses a cart / checkout workflow rather than exposing a simple direct file URL. |
| `usgs_v7` | USGS Spectral Library v7 | `https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae` | Preferred product is `ASCIIdata_splib07b_cvASD.zip` (“spectra convolved to ASD standard resolution”). The direct file URL is signed and expires, so it should be injected at build time rather than committed. |

## Metadata records only, with no downloadable file currently exposed

| source_id | name | record link | notes |
|---|---|---|---|
| `berlin_urban_materials_unavailable` | Berlin Urban Materials Spectral Library | `https://zenodo.org/records/11385631` | Zenodo record currently exposes `0` files. |
