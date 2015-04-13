Resilience calculator
====================

This program is intended to run the resilience indicator and draws most 
of the figures used in "Countries’ Socio-Economic Resilience to 
Natural Disasters" by Stephane Hallegatte, Mook Bangalore, and Adrien 
Vogt-Schilb. 
It comes with no warranty at all. 

Basic instructions: 

First all, you need a copy of ipython (find instructions for this part 
online). You have to know that I use python 3. The code won't run under 
python 2 without adjustments. 

To reproduce the figures in the paper 

1-download_wb_data.ipynb gets data from ASPIRE, WDI and FINDEX. 
2-comptue_res_ind.ipynb compiles the data and uses it to compute the 
resilience indicator. All the equations arein another file, 
res_ind_lib.py, and saves the results. 

After that, you can use: 
3-draw maps.ipynb to draw the maps. 
4-draw_plots.ipynb to draw... the plots. 

Drawing the scorecards requires 2 steps (also after steps 1 and 2): 
5-run compute_scorecards.ipynb This basically computes the derivative of 
risk and resilience wrt all the inputs 
6-run render_scorecards.ipynb. 
This does the actual drawing. 


===============================================================
Copyright (C) 2015  Adrien Vogt-Schilb

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.