# README

![Alt text](https://github.com/beaunorgeot/images_for_presentations/blob/master/logo_v4.tif?raw=true "annotation interface")

### What is PHIlter? 
The package and associated scripts provide an end-to-end pipeline for removing Protected Health Information from clinical notes (or other sensitive text documents) in a completely secure environment (a machine with no external connections or exposed ports). We use a combination of regular expressions, Part Of Speech (POS) and Entity Recognition (NER) tagging, and filtering through a whitelist to achieve nearly perfect Recall and generate clean, readable notes. Everything is written in straight python and the package will process any text file, regardless of structure. You can install with PIP (see below) and run with a single command-line argument. Parallelization of processesing can be infinitly divided across cores or machines. 

- Please note: we don't make any claims that running this software on your data will instantly produce HIPAA compliance. 



**philter:** Pull in each note from a directory (nested directories are supported), maintain clean words and replace PHI words with a safe filtered word: **\*\*PHI\*\***, then write the 'phi-reduced' output to a new file with the original file name appended by '\_phi\_reduced'. Conveniently generates additional output files containing meta data from the run: 
	- number of files processed
	- number of instances of PHI that were filtered
	- list of filtered words
	- etc.




# Installation

**Install philter**

```pip3 install philter```

### Dependencies
spacy package en: A pretrained model of the english language.
You can learn more about the model at the [spacy model documentation]("https://spacy.io/docs/usage/models") page. Language models are required for optimum performance. 

**Download the model:**

```python3 -m spacy download en ```

Note for Windows Users: This command must be run with admin previleges because it will create a *short-cut link* that let's you load a model by name.



# Run 

**philter**
``` philter -i ./raw_notes_dir -r -o dir_to_write_to -p 32```

Arguments:

- ("-i", "--input") = Path to the directory or the file that contains the PHI note, the default is ./input_test/
- ("-r", "--recursive") = help="whether read files in the input folder recursively. Default = False. 
- ("-o", "--output") = Path to the directory that save the PHI-reduced note, the default is ./output_test/.
- ("-w", "--whitelist") = Path to the whitelist, the default is phireducer/whitelist.pkl
- ("-n", "--name") = The key word of the output file name, the default is *_phi_reduced.txt.
- ("-p", "--process") = The number of processes to run simultaneously, the default is 1.


# How it works

**philter**

![Alt text](https://github.com/beaunorgeot/images_for_presentations/blob/master/flow_v2.tif?raw=true "phi-reduction process")


**Example Input and Output**
![Alt text](https://github.com/beaunorgeot/images_for_presentations/blob/master/deid_note_v2.tif?raw=true)




### Why did we build it?
Clinical notes capture rich information on the interaction between physicians, nurses, patients, and more. While this data holds the promise of uncovering valuable insights, it is also challenging to work with for numerous reasons. Extracting various forms of knowledge from Natural Language is difficult on it's own. However, attempts to even begin to mine this data on a large scale are severely hampered by the nature of the raw data, it's deeply personal. In order to allow more researchers to have access to this potentially transformative data, individual patient identifiers need to be removed in a way that presevers the content, context, and integrity of the raw note. 

De-Identification of clinical notes is certainly not a new topic, there are even machine learning competitions that are held to compare methods. Unfornuately these did not provide us with a viable approach to de-identify our own notes. First, the code from methods used in the competitions are often not available. 
Second, the notes used in public competitions don't reflect our notes very closely and therefore even methods that are publicly available did not perform nearly as well on our data as they did on the data used for the competitions (As noted by Ferrandez, 2012, BMC Medical Research Methodology who compared public methods on VA data). Additionally,our patient's privacy is paramount to us which meant we were unwilling to expose our data use any methods that required access to any url or external api call. Finally, our goal was to de-identify all 40 MILLION of our notes. There are multiple published approaches that are simply impractical from a run-time perspective at this scale. 

## Why a whitelist (aren't blacklists smaller and easier)?

Blacklists are certainly the norm, but they have some pretty large inherent problems. For starters, they present an unbounded problem: there are a nearly infinite number of words that could be PHI and that you'd therefore want to filter. For us, the difference between blacklists vs whitelists comes down to the *types* of errors that you're willing to make. Since blacklists are made of  PHI words and/or patterns, that means that when a mistake is made PHI is allowed through (Recall error). Whitelists on the other hand are made of non-PHI which means that when a mistake is made a non-PHI word gets filtered (Precision Error). We care more about recall for our own uses, and we think that high recall is also important to others that will use this software, so a whitelist was the sensible approach. 

### Results (current, unpublished)

![Alt text](https://github.com/beaunorgeot/images_for_presentations/blob/master/performance_1.png?raw=true "info_extraction_csv example")

# Recommendations
- Search through filtered words for institution specific words to improve precision
- have a policy in place to report phi-leakage