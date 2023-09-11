# Introduction 
This repo is for the dozerpush module which takes the positions of dozers during primary working time (PWT) in a given site, shift and shift description and clusters these, attributes properties such as track, speed, push/return/unclassified status; essentially this moduule returns dozer information or statistics for a shift. 

# Getting Started
To use this module, `pip install dozerpush` in databricks after installing the dozerpush library onto the cluster. Then to use the function that outputs the statistics and the statistics with the positions, call `dozer_push()`. The package, in its current state, takes the positional data of the production dozers as well as some tuning parameters. It is capable of working with multiple sites simultaneously.

For a given sihft, the site does not intervene with the logic deciding push/return/unclassified, the data is grouped by dozer. 

# Process
- Calc point grades, headings, speed, distance, duration
    - Calc distance as average forward and reverse point distance
    - Calc elevation delta for forward and reverse point
    - Calc time as average forward and reverse point time deltas
    - Calc 2nd order grade approximation from distance and elevation
    - Calc heading
- Identify straights
- Expand and extrapolate straights
- Extract straight stats
- Filter straight stats