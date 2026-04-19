# EMI Detection Project — Full End-to-End Project Phases

## Project Overview

The goal of this project is to build an end-to-end system that captures electromagnetic interference (EMI) signals from real devices, processes those signals into structured data, extracts meaningful representations from them, and uses machine learning to make predictions about device behavior or identity.

At a high level, the project flow is:

**Device → EMI Capture → Logged Signal Data → Clean Dataset → Processed Signals → Features → Models → Results → Demo**

Because this project combines hardware, signal processing, data engineering, and machine learning, it should be approached in structured phases. Each phase has a specific purpose, concrete deliverables, and technical outputs that feed into the next stage.

---

# Phase 1: Dataset Understanding, Audit, and Organization

## Purpose of This Phase

This phase is about understanding exactly what data exists, what each folder represents, how the files are structured, and what issues must be resolved before the dataset can be used reliably.

This is the foundation of the entire project. If the dataset is misunderstood, mislabeled, or ingested inconsistently, then every downstream step will be unreliable. The main job of this phase is to transform the raw hardware-delivered dataset into something the software team can reason about and use.

---

## Goals

- Identify every dataset folder and determine what experimental condition it represents
- Understand the meaning of each device folder, subfolder, and file type
- Audit the data for structural issues, inconsistencies, and missing information
- Define a canonical interpretation for labels such as device, state, distance, probe, and channel
- Build a clean metadata view of the dataset
- Prepare the raw data for systematic ingestion in code

---

## Key Questions This Phase Should Answer

- What devices are represented in the dataset?
- What does each top-level folder mean?
- Which files are time-domain signals and which are frequency-domain artifacts?
- Which folders are clean and ready for use, and which require cleanup?
- How are state and distance encoded?
- Which files are repeated captures or replicates?
- Are there sampling-rate inconsistencies?
- Are there mixed schemas in the same folder?
- Which parts of the dataset can be trusted as primary inputs?

---

## Technical Work

### 1. Folder and Condition Mapping
Every top-level folder should be interpreted as a specific measurement context. For example:
- charger 1 off
- charger 1 on close
- charger 2 off
- charger 2 on distant
- motor 1 on
- motor 2 on
- motor 3 on

The team should document what each folder means in plain language and in machine-readable form.

### 2. File Type Classification
Each file should be classified into one of the following:
- time-domain CSV
- frequency-domain CSV
- plot image
- note or metadata text file
- firmware/code file
- irrelevant artifact if any

### 3. Schema Inspection
For each CSV type, determine:
- header structure
- metadata row format
- sample row count
- channel layout
- whether the file is single-channel or dual-channel
- whether the file is valid or malformed

### 4. Data Integrity Audit
The dataset should be checked for:
- mixed time and frequency files in the same folder
- naming inconsistencies
- duplicate-style files such as `(1)` versions
- outlier sampling intervals
- broken FFT files
- missing labels or ambiguous folder meanings

### 5. Canonical Metadata Definition
The team should decide what metadata fields are required for each sample. For example:
- file path
- device family
- device ID
- state
- distance label
- channel mode
- increment
- sample count
- probe ID
- replicate ID
- quality flags

---

## Deliverables

### 1. Dataset Inventory Document
A structured document that explains:
- every top-level folder
- each subfolder type
- what each dataset represents
- known issues and warnings

### 2. Metadata Table
A file such as `dataset_index.csv` where each row represents one file and includes all known labels and data properties.

Example fields:
- `file_path`
- `device_family`
- `device_id`
- `state`
- `distance_label`
- `probe_id`
- `channel_mode`
- `increment_sec`
- `sample_count`
- `domain`
- `replicate_id`
- `quality_flags`

### 3. Dataset Cleanup Plan
A list of what needs to be fixed immediately, what can be isolated, and what can be deferred.

### 4. Ingestion Rules
A formal definition of how software should interpret each file type and condition.

---

## Success Criteria

This phase is complete when:
- the team can explain every major dataset folder confidently
- every usable file has a metadata row in a central index
- known dataset issues are identified and documented
- there is a clear plan for which data is primary and which data is secondary
- no one on the team is confused about what the folders mean

---

# Phase 2: Data Extraction and Ingestion Pipeline

## Purpose of This Phase

This phase turns raw CSV files into structured data objects that the rest of the pipeline can use. The dataset may be organized in folders, but models and signal processing code need structured arrays and metadata, not ad hoc file browsing.

The job of this phase is to create repeatable, automated ingestion code that reads raw files, extracts the signal, reconstructs timing information, and outputs a clean internal representation of the data.

---

## Goals

- Read every valid time-domain CSV automatically
- Extract waveform values from each file
- Reconstruct the time axis using metadata
- Detect and handle channel configurations
- Attach metadata to each extracted sample
- Create a consistent internal data representation for downstream teams

---

## Technical Work

### 1. CSV Parsing
The parser must:
- read header rows correctly
- identify where sample values begin
- handle single-channel and dual-channel layouts
- distinguish signal columns from metadata columns

Typical cases:
- `X, CH1, Start, Increment`
- `X, CH2, Start, Increment`
- `X, CH1, CH2, Start, Increment`

### 2. Signal Extraction
For each file:
- extract the sequence index
- extract one or more voltage channels
- store the signal as arrays
- validate that the sample count matches expectation

### 3. Time Axis Reconstruction
For time-domain files, physical time should be reconstructed using:
- `start_time`
- `increment`

For each sample index `i`, time can be computed as:
`time[i] = start + i * increment`

### 4. Channel Mode Handling
The pipeline must support:
- CH1-only data
- CH2-only data
- CH1 and CH2 dual-channel data

It should also record whether channel semantics are known, for example:
- CH1 = distant
- CH2 = close

### 5. Metadata Attachment
Each extracted sample should be paired with:
- source file path
- device labels
- state labels
- distance labels
- probe labels
- increment
- quality flags

### 6. Error Handling and Validation
The ingestion code should:
- skip malformed or unsupported files safely
- log warnings for quality issues
- isolate files marked for manual review
- avoid silently corrupting the dataset

---

## Deliverables

### 1. Data Loader Scripts
Code that can load raw CSV files and return structured outputs.

### 2. Parsed Signal Objects
A consistent format for all loaded signals. For example, each record might contain:
- signal array
- time array
- metadata dictionary

### 3. Validation Report
A summary of:
- number of files loaded successfully
- number of files skipped
- files with warnings
- files with mismatched or unusual metadata

### 4. Clean Ingestion Interface
A reusable API or script that other teams can call, such as:
- `load_file(...)`
- `load_folder(...)`
- `load_dataset(...)`

---

## Success Criteria

This phase is complete when:
- any valid file can be loaded automatically
- time-domain signals can be reconstructed without manual intervention
- metadata is attached consistently
- other teams can use the loader instead of manually inspecting files

---

# Phase 3: Signal Processing and Conditioning

## Purpose of This Phase

This phase improves signal quality and ensures that the waveforms are comparable across devices and recording conditions. Raw EMI captures often contain noise, scale differences, and inconsistencies that make direct analysis unreliable.

The goal here is to clean the signal without destroying the information that makes devices distinguishable.

---

## Goals

- reduce noise and irrelevant variation
- normalize the signal representation
- handle sampling-rate differences where necessary
- prepare the data for reliable feature extraction
- evaluate whether conditioning improves downstream separability

---

## Technical Work

### 1. Sampling Rate Management
Because the dataset uses more than one `Increment` value, the team must decide how to handle that difference.

Possible strategies:
- keep sampling rates separate and treat them as metadata
- resample signals onto a common time base
- compute features in a way that accounts for different sampling rates

This decision is important for FFT and frequency comparison.

### 2. Filtering
Possible filtering approaches may include:
- low-pass filtering
- band-pass filtering
- smoothing
- denoising techniques

The exact method should be selected based on signal behavior and project goals.

### 3. Normalization
Signals may need normalization so that comparisons are not dominated by amplitude scale differences.

Possible methods:
- z-score normalization
- min-max normalization
- RMS-based scaling

### 4. Signal Trimming or Windowing
If useful, signals may be windowed or segmented into regions of interest for more stable analysis.

### 5. Raw vs Processed Comparison
The team should compare:
- raw signals
- filtered signals
- normalized signals

This is necessary to justify that the conditioning step is helping rather than hiding useful structure.

---

## Deliverables

### 1. Processed Signal Pipeline
A repeatable procedure that converts raw signals into processed signals.

### 2. Filtered Signal Dataset
A saved set of processed signals or transformation scripts.

### 3. Conditioning Decision Document
A short write-up explaining:
- chosen filtering method
- chosen normalization method
- how sampling-rate differences were handled
- why those decisions were made

### 4. Comparative Visualizations
Plots showing:
- raw vs filtered signal
- before vs after normalization
- examples from different devices or states

---

## Success Criteria

This phase is complete when:
- processed signals are consistently cleaner and more comparable
- the conditioning pipeline is defined and reproducible
- downstream teams can rely on a stable input representation

---

# Phase 4: Feature Engineering

## Purpose of This Phase

This phase converts waveforms into compact, informative numerical representations that machine learning models can use. Rather than feeding raw signals directly into every model, it is often better in a project like this to extract interpretable signal features first.

This is where the team translates signal behavior into structured evidence.

---

## Goals

- transform processed signals into feature vectors
- compute both time-domain and frequency-domain representations
- build features that help distinguish devices, states, and distances
- support both single-channel and dual-channel datasets
- create a unified feature table for modeling

---

## Technical Work

### 1. Time-Domain Features
Possible features include:
- mean
- standard deviation
- variance
- RMS
- peak-to-peak amplitude
- max and min values
- zero-crossing rate
- signal energy
- skewness or kurtosis if useful

### 2. Frequency-Domain Features
Because provided FFT files are inconsistent, FFT should be recomputed from the time-domain signals in a standardized way.

Possible spectral features include:
- dominant frequency
- top frequency peaks
- spectral centroid
- spectral spread
- band energy
- total spectral energy
- harmonic structure indicators

### 3. Channel-Specific Features
For dual-channel data, features may be extracted:
- per channel
- as differences between channels
- as ratios between close and distant channels
- as joint summary features

### 4. Feature Table Construction
Each sample should be converted into one row of features with associated metadata labels.

This output may look like:
- file metadata columns
- time-domain feature columns
- frequency-domain feature columns

### 5. Feature Quality Assessment
The team should examine:
- which features vary meaningfully across classes
- whether certain features separate motors from chargers
- whether distance-related features show patterns
- whether some features are redundant

---

## Deliverables

### 1. Feature Extraction Code
A reproducible script or module that converts signals into features.

### 2. Feature Dataset
A file such as `features.csv` or a similar structured table.

### 3. Feature Dictionary
Documentation describing:
- what each feature means
- how each feature is computed
- whether it comes from time-domain or frequency-domain data

### 4. Exploratory Analysis
Basic plots or summaries showing which features appear most useful.

---

## Success Criteria

This phase is complete when:
- each usable sample can be converted into a consistent feature vector
- FFT is recomputed consistently from time-domain data
- there is a model-ready dataset with labels and features

---

# Phase 5: Modeling and Evaluation

## Purpose of This Phase

This phase uses the extracted features to train machine learning models that can predict meaningful labels from EMI signals. The objective is not just to get accuracy numbers, but to determine what the dataset can support reliably and what predictions are scientifically or practically meaningful.

---

## Goals

- define realistic prediction tasks
- build baseline models
- evaluate which tasks are reliable
- understand the limits of the dataset
- identify what makes signals distinguishable

---

## Possible Prediction Tasks

Depending on dataset quality, useful tasks may include:
- motor vs charger classification
- device ID classification
- charger-only ON vs OFF classification
- close vs distant classification for datasets where distance mapping is explicit

Some tasks are more reliable than others and should be chosen carefully.

---

## Technical Work

### 1. Problem Definition
The team must define:
- what is being predicted
- which subset of the data is being used
- which labels are trustworthy
- which evaluation metrics matter

### 2. Train-Test Split Strategy
The team must avoid leakage and unrealistic evaluation.

Possible considerations:
- separating replicate captures properly
- avoiding train/test contamination from repeated measurements
- ensuring fair splits across conditions

### 3. Baseline Model Development
Start with interpretable baseline models such as:
- logistic regression
- random forest
- support vector machine if helpful

The point is to establish a working baseline before attempting anything more complex.

### 4. Evaluation
Possible metrics include:
- accuracy
- precision
- recall
- F1 score
- confusion matrix

Evaluation should be tied to the specific task.

### 5. Error Analysis
The team should study:
- which classes are most confused
- whether errors are tied to distance, device variation, or noise
- whether certain devices dominate model behavior
- whether conditioning or features need revision

---

## Deliverables

### 1. Baseline Models
Working models for one or more prediction tasks.

### 2. Evaluation Report
A summary of:
- datasets used
- prediction targets
- metrics achieved
- limitations and caveats

### 3. Confusion Matrices and Performance Plots
Visual outputs showing how well the models performed.

### 4. Recommended Best Model
A final recommendation of which model and feature set should be used in the demo.

---

## Success Criteria

This phase is complete when:
- at least one meaningful prediction task works reliably
- results are quantified and documented
- the team understands both the strengths and limits of the model

---

# Phase 6: End-to-End Pipeline Integration

## Purpose of This Phase

This phase connects all independently built components into one coherent system. A project is not complete when there are separate scripts for cleaning, processing, features, and modeling; it is complete when those steps can run together in a defined order and produce predictions from raw inputs.

---

## Goals

- integrate all stages into one pipeline
- define input and output contracts between teams
- eliminate format mismatches
- create a repeatable end-to-end execution flow
- ensure the system is stable enough for demo use

---

## Technical Work

### 1. Interface Alignment
Each stage must agree on:
- file formats
- metadata field names
- channel conventions
- path conventions
- expected input and output structures

### 2. Pipeline Assembly
The system should support a path such as:
1. load raw file
2. parse signal and metadata
3. process or normalize signal
4. compute features
5. run trained model
6. return prediction

### 3. Debugging
During integration, the team should resolve:
- missing metadata fields
- incompatible feature columns
- sampling-rate handling mismatches
- dual-channel assumptions that do not generalize
- inconsistent file naming logic

### 4. Automation
Where possible, the team should implement scripts or notebooks that can run the full pipeline with minimal manual effort.

---

## Deliverables

### 1. End-to-End Workflow
A defined process that takes a raw measurement and returns a prediction.

### 2. Integration Scripts
Code that connects data loading, processing, feature extraction, and modeling.

### 3. Stable Input/Output Definitions
A documented contract for what each stage expects and returns.

### 4. Integration Test Cases
A small set of sample files that can be used to validate the full system.

---

## Success Criteria

This phase is complete when:
- the team can run the entire system from raw data to prediction
- all teams are using consistent interfaces
- the pipeline is stable enough to demonstrate repeatedly

---

# Phase 7: Results Analysis and Technical Interpretation

## Purpose of This Phase

This phase is about turning model outputs and engineering artifacts into understanding. It is not enough to say a model worked; the team should be able to explain what patterns were found, what signals were useful, what the limitations were, and how the hardware and software parts of the project connect.

---

## Goals

- interpret results in technical terms
- identify which features and conditions mattered most
- explain what the signals reveal about the devices
- connect engineering decisions to model performance
- prepare technical conclusions for the final presentation

---

## Technical Work

### 1. Comparative Analysis
Compare:
- motors vs chargers
- on vs off where valid
- close vs distant where valid
- raw vs processed signals
- single-channel vs dual-channel approaches

### 2. Feature Interpretation
Study which features appear most important or informative.

### 3. Result Contextualization
Discuss:
- what can actually be claimed based on the dataset
- where results may be confounded
- which outcomes are robust and which are exploratory

### 4. Hardware-Software Connection
Relate signal behavior back to:
- probe configuration
- distance effects
- sampling design
- EMI capture setup

---

## Deliverables

### 1. Results Summary
A concise explanation of:
- what worked
- what did not work
- why

### 2. Technical Insight Slides or Notes
Content that explains key results in a way suitable for presentation.

### 3. Visual Evidence
Plots such as:
- representative signal traces
- spectral comparisons
- confusion matrices
- feature distributions

---

## Success Criteria

This phase is complete when:
- the team can explain not just the results, but the meaning of the results
- the story of the project is technically coherent and evidence-based

---

# Phase 8: Demo Development

## Purpose of This Phase

This phase turns the project from a collection of analyses into something demonstrable. The demo should show the project’s core value clearly and simply. It does not need to be overly complicated; it just needs to be reliable, understandable, and connected to the project goals.

---

## Goals

- create a simple demonstration of the pipeline
- show how a signal becomes a prediction
- make the project tangible for an audience
- ensure the demo is stable and easy to explain

---

## Demo Options

Possible demos include:
- a script that takes an input file and prints predictions
- a notebook with visualizations and classification output
- a simple interface or dashboard showing waveform, features, and prediction

The choice should depend on time and team capacity.

---

## Technical Work

### 1. Demo Scope Definition
Select a focused use case, such as:
- motor vs charger prediction
- charger on vs off prediction
- close vs distant comparison for a valid subset

### 2. Input-Output Presentation
The demo should clearly show:
- input signal or selected sample
- preprocessing or feature summary
- predicted output
- confidence or explanation if available

### 3. Reliability Preparation
The demo should be tested on known good samples and protected against failure.

---

## Deliverables

### 1. Working Demo
A live or recorded demonstration of the project pipeline.

### 2. Demo Script or Instructions
Clear instructions for how to run the demo.

### 3. Demo Dataset
A small curated set of input examples for consistent presentation.

---

## Success Criteria

This phase is complete when:
- the project can be demonstrated clearly and reliably
- the audience can understand the core technical value quickly

---

# Phase 9: Final Presentation and Communication

## Purpose of This Phase

This phase packages the technical work into a clear, compelling story. The project presentation should explain the problem, the system, the challenges, the solutions, the results, and the final takeaway in a way that makes the work understandable and impressive.

---

## Goals

- present the project clearly from problem to result
- communicate both technical depth and practical outcome
- show teamwork across hardware and software
- explain challenges honestly and professionally
- highlight what the team built and learned

---

## Presentation Story Structure

A strong structure may look like:
1. problem and motivation
2. what EMI is and why it matters here
3. hardware capture process
4. dataset and data challenges
5. software pipeline
6. signal processing and feature engineering
7. modeling tasks and results
8. demo
9. limitations and future work

---

## Technical Communication Tasks

### 1. Slide Preparation
Slides should include:
- system diagrams
- cleaned pipeline overview
- key dataset facts
- representative signal plots
- feature or FFT visuals
- model results
- final conclusions

### 2. Speaker Preparation
Each team member should know:
- their role
- the technical decisions they contributed
- how their piece connects to the rest of the project

### 3. Story Refinement
The team should make sure the project story is:
- accurate
- concise
- technically grounded
- easy to follow

---

## Deliverables

### 1. Final Slide Deck
A polished presentation that communicates the whole project.

### 2. Speaking Plan
A clear division of speaking responsibilities.

### 3. Final Summary Statement
A concise statement of what the project achieved.

Example:
“We built an end-to-end system that captures EMI signals from devices, processes them into structured features, and uses machine learning to distinguish device behavior and identity.”

---

## Success Criteria

This phase is complete when:
- the team can present the project confidently
- the presentation explains both the engineering process and the final results
- the audience can clearly understand what was built and why it matters

---

# Final Project Outcome

By the end of all phases, the project should deliver:

- a clearly understood and documented EMI dataset
- a reliable ingestion and preprocessing pipeline
- processed and standardized signal data
- a feature extraction system based on time and spectral analysis
- baseline machine learning models for meaningful classification tasks
- an integrated raw-to-prediction workflow
- technical results supported by visual evidence
- a working demo
- a polished final presentation

The ideal final outcome is not just “we trained a model,” but:

**we built a complete system that converts real-world EMI measurements into structured predictions through data engineering, signal processing, and machine learning.**