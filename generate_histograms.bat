@echo off
echo ================================================================================
echo  Generating Single-Color Histograms for SPECK Encryption  
echo  (Deleting old RGB histograms and creating new grayscale ones)
echo ================================================================================
echo.
python compare_and_plot_histograms.py
echo.
echo ================================================================================
echo  Complete! Check the histograms folder for new single-color histograms
echo ================================================================================
pause
