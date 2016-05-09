Resilience calculator
====================

This program is intended to run the resilience indicator and draw most of the figures used in "Assessing Socioeconomic Resilience to Floods in 90 Countries" World Bank Policy Research Working Paper by Stephane Hallegatte, Mook Bangalore, and Adrien Vogt-Schilb, 2016. 
It comes with no warranty at all. 


Basic instructions: 

First all, you need a copy of python 3.x and the jupyter notebook (find instructions for this part online). 
To reproduce the figures in the paper :

* download_wb_data.ipynb gets data from ASPIRE, WDI and FINDEX. However, some of the data used in the paper has been removed from ASPIRE since the first time we downloaded it (share of income from remittances). So instead of downloading the data, use inputs/wb_data_backup.csv.

* compute_res_ind.ipynb compiles the data, uses it to compute the resilience at the contry level, and saves the results. All the equations are in another file, res_ind_lib.py. 

After that, you can use:

* draw maps.ipynb to draw the maps. (This script may use ImageMagick and Inkspace. Find them online. Windows users: add them to your PATH)
* draw_plots.ipynb to draw the plots. 
* mumbain_resilience_and_sensitivity : Mumbai case study

Drawing the scorecards requires 2 steps (also after two first steps): 

* run compute_scorecards.ipynb. This basically computes the derivative of risk and resilience wrt all the inputs.
* run render_scorecards.ipynb. This does the actual drawing of the scorecards. This script will also attempt to convert all the scorecards from eps to png and to a single pdf. It uses ImageMagick and Ghostscript (find them online).

Other files:
* Compare indices.ipynb : compares our results to other resilience indicators



===============================================================
Copyright (C) 2015--2016  Adrien Vogt-Schilb

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

===============================================================

