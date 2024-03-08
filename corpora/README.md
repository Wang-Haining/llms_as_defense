# Corpora

This directory hosts the corpora used in the experiments.

## The Riddell-Juola Corpus (RJ)

### Source
Commit `4fb89058ef` from 
https://codeberg.org/lab2124-sandbox/defending-against-authorship-attribution-corpus/src/branch/master/corporus.
MD5 `c13b4218e498234ff058d3a0154e1c2d` for defending-against-authorship-attribution-corpus-master.zip.

The folder 'corpus' has been renamed to 'rj'. 

### Notes
[todo] xx does not have xxx, so we remove it everywhere.

## The Extended Brennan-Greenstadt Corpus (EBG)

### Source
Commit `08d4758` from https://github.com/psal/jstylo/tree/master/jsan_resources/corpora/amt.
MD5 `4b51153cd4f64678d19e8d7b638345a2` for jstylo-master.zip.
The folder 'amt' has been renamed to 'ebg'. 

### Notes
Plain text (.txt) files ending with '_verification.txt' and '_demographics.txt' are not used.
If an author is associated with more than one '_imitation*.txt', only '_imitation_01.txt' is used.
[todo] A random sample is held out as the test.

## Loyola Computer Mediated Communication Corpus (LCMC) v1.1

### Source
MD5 `ef06a4201007398b7d30f1a3e7d6b2e0` for CMCCData.zip
Data files were obtained through personal communication with [Roberta Sabin](res@loyola.edu).
Paper: Goldstein, J., Winder, R., & Sabin, R. (2009, March). Person identification from text and speech genre samples. 
In Proceedings of the 12th Conference of the European Chapter of the ACL (EACL 2009) (pp. 336-344).

Two notable coding errors were fixed:
- 'S21S2G4.txt' -> 'S21D2G4.txt'.
- 'S1D113.txt' -> 'S1D1I3.txt'.
We therefore versioned this corpus as v1.1.


## IMDb62

### Source
`https://umlt.infotech.monash.edu/?page_id=266`
MD5 `e75089d9a050e6e119d4989d591ec900` for imdb62.zip.


## Blog1000

### Source
v1.1 from https://doi.org/10.5281/zenodo.7455623
MD5 `0a9e38740af9f921b6316b7f400acf06` for blog1000.csv.gz.


