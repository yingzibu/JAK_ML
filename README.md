# JAK_ML
Data were extracted from pubchem, bindingDB, and ChemBL. Aim is to build separate model for JAK1, JAK2, JAK3, and TYK2 to predict inhibitors.
raw data was handled: 
  * Active: 1
  * Inactive: 0
  * Unspecifed: -1
  * Inconclusive: deleted, not in file anymore
Further, the unspecifed -1 was merged with Inactive 0 as there were much fewer inactive drugs. 

FLT3 were also extracted from pubchem. Since nonspecific targeting to FLT3 is related with GI toxicity to JAK inhibitors, we may also build a model to predict compounds on FLT3 inhibition, and would like to avoid FLT3 inhibition to lower GI toxicity for drugs targeting IBD (GI tract diseases)
