## CONVOCALS: a CONVOlutional neural network to predict symptoms and major secondary CArdiovascuLar events based on high-resolution scanned histological Slides.

[![DOI]()]()

<!-- Please add a brief introduction to explain what the project is about    -->
Francesco Cisternino<sup>5\*</sup>, Yipei Song<sup>1\*</sup>, Gert Jan de Borst<sup>2</sup>, Joost Mekke<sup>2</sup>, Barend Mol<sup>2</sup>, Dominique P.V. de Kleijn<sup>2</sup>, Gerard Pasterkamp<sup>3</sup>, Aryan Vink<sup>4</sup>, Sander W. van der Laan<sup>3\*</sup>, Clint L. Miller<sup>1\*</sup>, Craig Glastonbury<sup>5\*</sup>. \* Authors contributed equally.

<sup>1) Center for Public Health Genomics, Department of Public Health Sciences, Department of Biochemistry and Molecular Genetics, University of Virginia, Charlottesville, VA 22908, USA. 2) Department of Vascular Surgery, Division Surgical Specialties, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands. 3) Central Diagnostics Laboratory, Division Laboratories, Pharmacy, and Biomedical Genetics, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands. 4) Department of Pathology, Division Laboratories, Pharmacy, and Biomedical Genetics, University Medical Center Utrecht, Utrecht University, Utrecht, the Netherlands. 5) Human Technopole, Viale Rita Levi-Montalcini, 1, 20157, Milano, Italy.</sup>


### Background

Despite tremendous medical progress, cardiovascular diseases (CVD) are still topping global charts of morbidity and mortality. Atherosclerosis is the major underlying cause of CVD and results in atherosclerotic plaque formation. The extent and type of atherosclerosis is manually assessed through histological analysis, and histological characteristics are linked to major acute cardiovascular events (MACE). However, conventional means of assessing plaque characteristics suffer major limitations directly impacting their predictive power. CONVOCALS will use a machine learning technique, convolutional neural network (CNN), to develop an internal representation of the 2-dimensional plaque images, allowing the model to learn position and scale in variant structures in the data.  A CNN is a subset of deep learning which has established as a powerful class of models for image recognition problems such as analysis of x-ray medical images. The aim of CONVOCALS is to build a CNN to process high-resolution images from scanned histological slides of plaques in order to predict MACE. 


#### Study design

We will use data from the [*Athero-Express Biobank Study (AE)*](https://doi.org/10.1007/s10564-004-2304-6){target="_blank"} comprising ±2,500 carotid endarterectomy patients of whom extensive clinical data (demographic, lifestyle, laboratory, medical history, and medication) as well as plaques are collected. At two Dutch tertiary referral centers patients are included that underwent endarterectomy; details of the study design were described [before](https://doi.org/10.1007/s10564-004-2304-6){target="_blank"}. Briefly, blood and plaque material were obtained during endarterectomy and stored at -80 ℃. Only carotid endarterectomy (CEA) patients were included in the present study. All patients provided informed consent and the study was approved by the medical ethics committee.
All plaques are histological assessed using 9 different standardized protocols for CD34, CD66b, CD68, SMA, elastin, hematoxylin, picro-sirius red, fibrin, glycophorin C and scanned at high-resolution into `.ndpi` or `.TIF` whole-slide images (WSI). For CONVOCALS we will use all the available data, i.e. ± 22,500 images and clinical data, to build a CNN using advanced computer algorithms as implemented in Python and classify patients based on 1) symptoms, and 2) MACE.


<!-- Please add a brief introduction to explain what the project is about    -->

### Where do I start?

You can load this project in RStudio by opening the file called 'CONVOCALS.Rproj'.

### Project structure

<!--  You can add rows to this table, using "|" to separate columns.         -->
File                | Description                | Usage         
------------------- | -------------------------- | --------------
README.md           | Description of project     | Human editable
CONVOCALS.Rproj     | Project file               | Loads project 
LICENSE             | User permissions           | Read only     
.worcs              | WORCS metadata YAML        | Read only     
renv.lock           | Reproducible R environment | Read only     
images              | Images used in readme, etc | Human editable
scripts             | Script to process data     | Human editable

<!--  You can consider adding the following to this file:                    -->
<!--  * A citation reference for your project                                -->
<!--  * Contact information for questions/comments                           -->
<!--  * How people can offer to contribute to the project                    -->
<!--  * A contributor code of conduct, https://www.contributor-covenant.org/ -->

### Reproducibility

This project uses the Workflow for Open Reproducible Code in Science (WORCS) to
ensure transparency and reproducibility. The workflow is designed to meet the
principles of Open Science throughout a research project. 

To learn how WORCS helps researchers meet the TOP-guidelines and FAIR principles,
read the preprint at https://osf.io/zcvbs/

#### WORCS: Advice for authors

* To get started with `worcs`, see the [setup vignette](https://cjvanlissa.github.io/worcs/articles/setup.html)
* For detailed information about the steps of the WORCS workflow, see the [workflow vignette](https://cjvanlissa.github.io/worcs/articles/workflow.html)

#### WORCS: Advice for readers

Please refer to the vignette on [reproducing a WORCS project]() for step by step advice.
<!-- If your project deviates from the steps outlined in the vignette on     -->
<!-- reproducing a WORCS project, please provide your own advice for         -->
<!-- readers here.                                                           -->

### Questions and issues

<!-- Do you have burning questions or do you want to discuss usage with other users? Please use the Discussions tab.-->

Do you have burning questions or do you want to discuss usage with other users? Do you want to report an issue? Or do you have an idea for improvement or adding new features to our method and tool? Please use the [Issues tab](https://github.com/CirculatoryHealth/EntropyMasker/issues).


### Citations

Using our **`EntropyMasker`** method? Please cite our work:

    An automatic entropy method to efficiently mask histology whole-slide images
    Yipei Song, Francesco Cisternino, Joost Mekke, Gert Jan de Borst, Dominique P.V. de Kleijn, Gerard Pasterkamp, Aryan Vink, Craig Glastonbury, Sander W. van der Laan, Clint L. Miller.
    medRxiv 2022.09.01.22279487; doi: https://doi.org/10.1101/2022.09.01.22279487


### Data availability

The whole-slide images used in this project are available through a [DataverseNL repository](https://doi.org/10.34894/QI135J "ExpressScan: Histological whole-slide image data from the Athero-Express (AE) and Aneurysm-Express (AAA) Biobank Studies"). There are restrictions on use by commercial parties, and on sharing openly based on (inter)national laws, regulations and the written informed consent. Therefore these data (and additional clinical data) are only available upon discussion and signing a Data Sharing Agreement (see Terms of Access) and within a specially designed UMC Utrecht provided environment.

### Acknowledgements

We are thankful for the support of the Netherlands CardioVascular Research Initiative of the Netherlands Heart Foundation (CVON 2011/B019 and CVON 2017-20: Generating the best evidence-based pharmaceutical targets for atherosclerosis [GENIUS I&II]), the ERA-CVD program 'druggable-MI-targets' (grant number: 01KL1802), and the Leducq Fondation 'PlaqOmics'.

Funding for this research was provided by National Institutes of Health (NIH) grant nos. R00HL125912 and R01HL14823 (to Clint L. Miller), and a Leducq Foundation Transatlantic Network of Excellence ('PlaqOmics') grant no. 18CVD02 (to Dr. Clint L. Miller and Dr. Sander W. van der Laan), and EU H2020 TO_AITION grant no. 848146 (to Dr. Sander W. van der Laan).

Dr. Sander W. van der Laan has received Roche funding for unrelated work.

Dr Craig A. Glastonbury has stock options in BenevolentAI and is a paid consultant for BenevolentAI, unrelated to this work.

Plaque samples are derived from arterial endarterectomies as part of the [Athero-Express Biobank Study](https://doi.org/10.1007/s10564-004-2304-6) which is an ongoing study in the UMC Utrecht. We would like to thank all the (former) employees involved in the Athero-Express Biobank Study of the Departments of Surgery of the St. Antonius Hospital Nieuwegein and University Medical Center Utrecht for their continuing work. In particular we would like to thank (in no particular order) Marijke Linschoten, Arjan Samani, Petra H. Homoed-van der Kraak, Tim Bezemer, Tim van de Kerkhof, Joyce Vrijenhoek, Evelyn Velema, Ben van Middelaar, Sander Reukema, Robin Reijers, Joëlle van Bennekom, and Bas Nelissen. Lastly, we would like to thank all participants of the Athero-Express Biobank Study; without you these studies would not be possible.

The framework was based on the [`WORCS` package](https://osf.io/zcvbs/).

<center><a href='https://www.era-cvd.eu'><img src="images/ERA_CVD_Logo_CMYK.png" align="center" height="75"/></a> <a href='https://www.to-aition.eu'><img src="images/to_aition.png" align="center" height="75"/></a> <a href='https://www.plaqomics.com'><img src="images/leducq-logo-large.png" align="center" height="75"/></a> <a href='https://www.fondationleducq.org'><img src="images/leducq-logo-small.png" align="center" height="75"/></a> <a href='https://osf.io/zcvbs/'><img src="images/worcs_icon.png" align="center" height="75"/></a> <a href='https://www.atheroexpress.nl'><img src="images/AE_Genomics_2010.png" align="center" height="100"/></a></center>

#### Changes log

    Version:      v1.0.1
    Last update:  2022-10-04
    Written by:   Francesco Cisternino; Craig Glastonbury; Sander W. van der Laan; Clint L. Miller; Yipei Song.
    Description:  CONVOCALS repository: classification of atherosclerotic histological whole-slide images
    Minimum requirements: R version 3.4.3 (2017-06-30) -- 'Single Candle', Mac OS X El Capitan

    **MoSCoW To-Do List**
    The things we Must, Should, Could, and Would have given the time we have.
    _M_

    _S_

    _C_

    _W_
    
    Changes log
    * v1.0.1 Updates and re-organization.
    * v1.0.0 Initial version. 
    
    
--------------

#### Creative Commons BY-NC-ND 4.0
##### Copyright (c) 2022 [Francesco Cisternino]() \| [Craig Glastonbury](https://github.com/GlastonburyC) \| [Sander W. van der Laan](https://github.com/swvanderlaan) \| [Clint L. Miller](https://github.com/clintmil) \| [Yipei Song](https://github.com/PetraSong) 

<sup>This is a human-readable summary of (and not a substitute for) the [license](LICENSE). </sup>
</br>
<sup>You are free to share, copy and redistribute the material in any medium or format. The licencor cannot revoke these freedoms as long as you follow the license terms.</br></sup>
</br>
<sup>Under the following terms: </br></sup>
<sup><em>- Attribution</em> — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.</br></sup>
<sup><em>- NonCommercial</em> — You may not use the material for commercial purposes.</br></sup>
<sup><em>- NoDerivatives</em> — If you remix, transform, or build upon the material, you may not distribute the modified material.</br></sup>
<sup><em>- No additional</em> restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.</br></sup>
</br></sup>
<sup>Notices: </br></sup>
<sup>You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.</sup>


