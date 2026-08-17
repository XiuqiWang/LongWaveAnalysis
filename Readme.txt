To launch spyder in the kg2 environment:
On Anaconda Prompt,
conda activate kg2
spyder

Autospectrum:
Velocity measurements were rotated into cross-shore and alongshore components, where the cross-shore direction was defined by the transect connecting Frames F1 and F3. Pressure data that have been corrected for the atmospheric pressure were converted to free surface elevation using linear wave theory. The velocity and surface-elevation records were divided into natural bursts of approximately 30 min, quadratically detrended, and processed using Welch's method to estimate autospectra, using 20 Hann-windowed segments with 50% overlap. Spectral variance was then integrated over the infragravity (0.005–0.05 Hz) and sea-swell (0.05–1 Hz) frequency bands to obtain the corresponding spectral wave heights. For the surface-elevation spectra, the upper limit of the sea-swell band was truncated at the frequency where the required pressure-correction amplification factor exceeded 10, substantially more conservative than the amplification factors of 100–1000 discussed by Bishop and Donelan (1987).

DOF of spectrum calculation: K = 1+(N-Nseg)/Nstep, N = fs*30*1800, Nseg = fs*512, Nstep = Nseg - Noverlap, Noverlap = 50%*Nseg
                             DOF = 1.5-2 * K 
			     DOF = 9-12
                             
                             common DOF 10-40
                             frequency resolution = 1/Tseg
 
Spectrum per block: wave condition should be stationary per spectrum; saving the spectra per block can store the variation from calm to stormy conditions

Remaining question:
Longshore velocity variance is higher than cross-shore. 
