classdef DipTab_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        DipTabUIFigure                  matlab.ui.Figure
        SampleDescriptionEditField      matlab.ui.control.EditField
        DescriptionEditFieldLabel       matlab.ui.control.Label
        DeployButton                    matlab.ui.control.Button
        DipTabLabel                     matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        StopProcessButton               matlab.ui.control.Button
        TabGroup                        matlab.ui.container.TabGroup
        TimeDomainTab                   matlab.ui.container.Tab
        LoadRawDataButton               matlab.ui.control.Button
        KernelSmoothingPanel            matlab.ui.container.Panel
        gausswinEditField               matlab.ui.control.NumericEditField
        gausswinEditFieldLabel          matlab.ui.control.Label
        ApplyButton                     matlab.ui.control.Button
        LiquidfrontExtractionButton     matlab.ui.control.Button
        SetDownLimitButton              matlab.ui.control.Button
        SetLeftLimitButton              matlab.ui.control.Button
        XlinesecLabel                   matlab.ui.control.Label
        tYPickEditField                 matlab.ui.control.NumericEditField
        tYPickSlider                    matlab.ui.control.Slider
        tXPickEditField                 matlab.ui.control.NumericEditField
        YlinesecLabel                   matlab.ui.control.Label
        tXPickSlider                    matlab.ui.control.Slider
        SaveFigureButton                matlab.ui.control.Button
        ColormapcontrolPanel_TD         matlab.ui.container.Panel
        AOIRangeToEditField             matlab.ui.control.NumericEditField
        Label_2                         matlab.ui.control.Label
        AOIRangeFromEditField           matlab.ui.control.NumericEditField
        DOIRangeLabel                   matlab.ui.control.Label
        DataRangeToEditField            matlab.ui.control.NumericEditField
        Label                           matlab.ui.control.Label
        DataRangeFromEditField          matlab.ui.control.NumericEditField
        DataRangeEditFieldLabel         matlab.ui.control.Label
        SaveTruncatedButton             matlab.ui.control.Button
        FrequencyDomainStudyButton      matlab.ui.control.Button
        AOIBoundaryTruncationPanel      matlab.ui.container.Panel
        DownEditField                   matlab.ui.control.NumericEditField
        DownEditFieldLabel              matlab.ui.control.Label
        UpEditField                     matlab.ui.control.NumericEditField
        UpEditFieldLabel                matlab.ui.control.Label
        RightEditField                  matlab.ui.control.NumericEditField
        RightEditFieldLabel             matlab.ui.control.Label
        LeftEditField                   matlab.ui.control.NumericEditField
        LeftEditFieldLabel              matlab.ui.control.Label
        RemoveBaseButton_Tab1           matlab.ui.control.Button
        TruncateButton                  matlab.ui.control.Button
        ColormapPanel                   matlab.ui.container.Panel
        DCheckBox                       matlab.ui.control.CheckBox
        colorbarCheckBox                matlab.ui.control.CheckBox
        PlotButton                      matlab.ui.control.Button
        ColormapDropDown                matlab.ui.control.DropDown
        GuidelinesPanel                 matlab.ui.container.Panel
        AutoScanSpectrumCheckBox        matlab.ui.control.CheckBox
        PowerSpectrumButton             matlab.ui.control.Button
        SpectrogramButton               matlab.ui.control.Button
        EnableButton                    matlab.ui.control.StateButton
        GeneralInformationPanel         matlab.ui.container.Panel
        RefractiveIndexEditField        matlab.ui.control.NumericEditField
        RefractiveIndexEditFieldLabel   matlab.ui.control.Label
        ThicknessmmEditField            matlab.ui.control.NumericEditField
        ThicknessmmEditFieldLabel       matlab.ui.control.Label
        TimeSpacingsEditField           matlab.ui.control.NumericEditField
        TimeSpacingsEditFieldLabel      matlab.ui.control.Label
        ToFSpacingpsEditField           matlab.ui.control.NumericEditField
        XSpacingpsEditFieldLabel        matlab.ui.control.Label
        DataNumberEditField             matlab.ui.control.NumericEditField
        DataNumberLabel                 matlab.ui.control.Label
        DataLengthEditField             matlab.ui.control.NumericEditField
        DataLengthLabel                 matlab.ui.control.Label
        UIAxesTD4                       matlab.ui.control.UIAxes
        UIAxesTD3                       matlab.ui.control.UIAxes
        UIAxesTD2                       matlab.ui.control.UIAxes
        UIAxesTD1                       matlab.ui.control.UIAxes
        LiquidfrontExtractionTab        matlab.ui.container.Tab
        SaveFigureButton_LE             matlab.ui.control.Button
        AutoSaveFiguresCheckBox         matlab.ui.control.CheckBox
        SampleNameEditField             matlab.ui.control.EditField
        SampleNameEditFieldLabel        matlab.ui.control.Label
        LinearFittingPanel              matlab.ui.container.Panel
        RMSEEditField                   matlab.ui.control.NumericEditField
        RMSEEditFieldLabel              matlab.ui.control.Label
        R2EditField                     matlab.ui.control.NumericEditField
        R2EditFieldLabel                matlab.ui.control.Label
        FittingFunctionktdLabel         matlab.ui.control.Label
        dmmEditField                    matlab.ui.control.NumericEditField
        dmmEditFieldLabel               matlab.ui.control.Label
        kmmsEditField                   matlab.ui.control.NumericEditField
        kmmsEditFieldLabel              matlab.ui.control.Label
        CaculateFittingParametersButton  matlab.ui.control.Button
        LiquidIngressTimesecEditField   matlab.ui.control.NumericEditField
        LiquidIngressTimesecEditFieldLabel  matlab.ui.control.Label
        GeneralInformationPanel_LE      matlab.ui.container.Panel
        RefractiveIndexEditField_LE     matlab.ui.control.NumericEditField
        n_effLabel                      matlab.ui.control.Label
        ThicknessmmEditField_LE         matlab.ui.control.NumericEditField
        ThicknessmmEditField_2Label     matlab.ui.control.Label
        dataNumberEditField_LE          matlab.ui.control.NumericEditField
        NumberofScansEditField_2Label_2  matlab.ui.control.Label
        dataLengthEditField_LE          matlab.ui.control.NumericEditField
        DataLengthEditField_2Label_2    matlab.ui.control.Label
        DataComparisonButton            matlab.ui.control.Button
        ROISelectionPanel               matlab.ui.container.Panel
        Upto3mmCheckBox                 matlab.ui.control.CheckBox
        DisplayTabletCentreCheckBox     matlab.ui.control.CheckBox
        ROIwidthEditField               matlab.ui.control.NumericEditField
        ROIwidthEditFieldLabel          matlab.ui.control.Label
        DrawPolylineButton              matlab.ui.control.Button
        LfPlotButton                    matlab.ui.control.Button
        ExtColormapDropDown             matlab.ui.control.DropDown
        AlphaDropDown                   matlab.ui.control.DropDown
        AlphaDropDown_2Label            matlab.ui.control.Label
        UIAxesLE3                       matlab.ui.control.UIAxes
        UIAxesLE2                       matlab.ui.control.UIAxes
        UIAxesLE1                       matlab.ui.control.UIAxes
        DataComparisonTab               matlab.ui.container.Tab
        DownButton                      matlab.ui.control.Button
        UpButton                        matlab.ui.control.Button
        ExportTableasCSVFormatButton    matlab.ui.control.Button
        PlotButton_DCNew                matlab.ui.control.Button
        PlotButton_DC3D                 matlab.ui.control.Button
        PlotButton_DC                   matlab.ui.control.Button
        LegendCheckBox                  matlab.ui.control.CheckBox
        PlotButtonGroup                 matlab.ui.container.ButtonGroup
        BatchButton                     matlab.ui.control.RadioButton
        IndividualButton                matlab.ui.control.RadioButton
        FittingFunctionParametersLabel  matlab.ui.control.Label
        UITable                         matlab.ui.control.Table
        LoadProjectButton               matlab.ui.control.Button
        SaveProjectButton               matlab.ui.control.Button
        AssigndatainworkspaceButton     matlab.ui.control.Button
        AddButton                       matlab.ui.control.Button
        RemoveButton_DC1                matlab.ui.control.Button
        BatchDetailListBox              matlab.ui.control.ListBox
        BatchDetailListBoxLabel         matlab.ui.control.Label
        BatchListBox                    matlab.ui.control.ListBox
        BatchListBoxLabel               matlab.ui.control.Label
        BatchNameEditField              matlab.ui.control.EditField
        BatchNameEditFieldLabel         matlab.ui.control.Label
        MeasurementListBox              matlab.ui.control.ListBox
        MeasurementListBoxLabel         matlab.ui.control.Label
        RemoveButton_DC2                matlab.ui.control.Button
        UngroupButton                   matlab.ui.control.Button
        GroupButton                     matlab.ui.control.Button
        UIAxesCD1                       matlab.ui.control.UIAxes
        UIAxesCD2                       matlab.ui.control.UIAxes
        FrequencyDomainTab              matlab.ui.container.Tab
        ColormapcontrolPanel            matlab.ui.container.Panel
        ftColorbarCheckBox              matlab.ui.control.CheckBox
        ftColormapDropDown              matlab.ui.control.DropDown
        ColormapDropDown_4Label         matlab.ui.control.Label
        AOIRangeToEditField_2           matlab.ui.control.NumericEditField
        Label_4                         matlab.ui.control.Label
        AOIRangeFromEditField_2         matlab.ui.control.NumericEditField
        DOIRangeLabel_2                 matlab.ui.control.Label
        DataRangeToEditField_2          matlab.ui.control.NumericEditField
        Label_3                         matlab.ui.control.Label
        DataRangeFromEditField_2        matlab.ui.control.NumericEditField
        DataRangeEditFieldLabel_2       matlab.ui.control.Label
        PlotButton_3                    matlab.ui.control.Button
        AssiginFFTDatainworkspaceButton  matlab.ui.control.Button
        SingleMeasurementPanel          matlab.ui.container.Panel
        LocationSlider                  matlab.ui.control.Slider
        LocationSliderLabel             matlab.ui.control.Label
        AutoScanButton                  matlab.ui.control.Button
        EnableButton_2                  matlab.ui.control.StateButton
        NextButton                      matlab.ui.control.Button
        DataInformationPanel            matlab.ui.container.Panel
        dataNumberEditField_FD          matlab.ui.control.NumericEditField
        NumberofScansEditField_2Label   matlab.ui.control.Label
        dataLengthEditField_FD          matlab.ui.control.NumericEditField
        DataLengthEditField_2Label      matlab.ui.control.Label
        FourierTransformPanel           matlab.ui.container.Panel
        FunctionDropDown                matlab.ui.control.DropDown
        FunctionDropDownLabel           matlab.ui.control.Label
        ToEpolFreqEditField             matlab.ui.control.NumericEditField
        toLabel_2                       matlab.ui.control.Label
        FromEpolFreqEditField           matlab.ui.control.NumericEditField
        fromLabel_2                     matlab.ui.control.Label
        ExtrapolationRangeTHzLabel      matlab.ui.control.Label
        StartFrequencyTHzEditField      matlab.ui.control.NumericEditField
        StartFrequencyTHzEditFieldLabel  matlab.ui.control.Label
        UnwrappingLabel                 matlab.ui.control.Label
        ZeroFillingpowerofSpinner       matlab.ui.control.Spinner
        ZeroFillingpowerofSpinnerLabel  matlab.ui.control.Label
        UpsamplingLabel                 matlab.ui.control.Label
        ToFreqEditField                 matlab.ui.control.NumericEditField
        toLabel                         matlab.ui.control.Label
        FromFreqEditField               matlab.ui.control.NumericEditField
        fromLabel                       matlab.ui.control.Label
        FrequencyRangeTHzLabel          matlab.ui.control.Label
        TransformButton                 matlab.ui.control.Button
        UIAxesFD4                       matlab.ui.control.UIAxes
        UIAxesFD3                       matlab.ui.control.UIAxes
        UIAxesFD2                       matlab.ui.control.UIAxes
        UIAxesFD1                       matlab.ui.control.UIAxes
        SystemStatusEditField           matlab.ui.control.EditField
        SystemStatusEditFieldLabel      matlab.ui.control.Label
        ProjectNameEditField            matlab.ui.control.EditField
        TerahertzLqiuidFrontDateAnalyserLabel  matlab.ui.control.Label
        ImportthzFileButton             matlab.ui.control.Button
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The dotTHz project, 2023 TAG, University of Cambridge
% Terahertz Liquid Front Transport Analysis Software
% In case of any question or enqury please contact the developer,
% Jongmin Lee at jl2112@cam.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        
    properties (Access = private)
        filefullpath % imported project file full path
        TData % Time domain data (measurement)
        FData % Frequency domain data (fast Fourier transfrom)
        stopProcess % variable for forcing process stopped     
        Handler % axis component handler
        CData % Data comparison data
        plotUpdate % display update option
        maxSam % max sample value
        inAng = deg2rad(8.8) % THz beam incident angle
        Tcell % Cell structure for table
    end
    
    methods (Access = private)
        
        function resetGeneralInfo(app)
            app.DataLengthEditField.Value = 0;
            app.DataNumberEditField.Value = 0; 
            app.LeftEditField.Value = 0;
            app.RightEditField.Value = 0;
            app.UpEditField.Value = 0;
            app.DownEditField.Value = 0;
        end
        
        function TDPlot(app,autoStr)
            samData = app.TData.samData;
            ToF = app.TData.ToF;
            xData = app.TData.xData;
            app.maxSam = max(samData,[],"all");
            
            ax1 = app.UIAxesTD1;
            ax3 = app.UIAxesTD3;
            app.stopProcess = 0;
            app.EnableButton.Value = false;
            
            clo(ax1)
            axis(ax1,"tight")
            cmap = app.ColormapDropDown.Value;
            aoiMin = app.AOIRangeFromEditField.Value;
            aoiMax = app.AOIRangeToEditField.Value;
            crange = [aoiMin aoiMax];
            clim(ax1,crange);
            colormap(ax1,cmap)

            ylabel(ax1,'Time of flight (ps)');
            
            if app.colorbarCheckBox.Value
                colorbar(ax1)
            else
                colorbar(ax1,"off")
            end
            
            app.SystemStatusEditField.Value = 'Displaying...';
            drawnow;
            
            if app.DCheckBox.Value % 3D checkbox
                surf(ax1,xData,ToF,samData,"EdgeColor","none");
                view(ax1,2);
            else                
                imagesc(ax1,xData,ToF,samData);
                %ax1.YDir = 'normal';
            end
            
            grid(ax3,"on")
            edfSum = sum(abs(samData));
            plot(ax3,xData,edfSum);
            axis(ax3,"tight")
            % xlim(ax3,"tight");

            app.SystemStatusEditField.Value = 'Done';           
        end
        
        function posT1_XYLine(app)
            xPick = app.tXPickSlider.Value;
            yPick = app.tYPickSlider.Value;

            xData = app.TData.xData;
            ToF = app.TData.ToF;
            maxSam = app.maxSam;
            ax2 = app.UIAxesTD2;

            grid(ax2,"on");
            
            app.Handler.xline.Value = xPick;
            app.Handler.CExline.Value = xPick;
            app.Handler.yline.Value = yPick;
            
            cla(ax2)
                    
            % plot THz signal of the cross pointed location
            loc = sum(xData<=xPick);
            app.TData.xPick = loc;
            samEfd = app.TData.samData(:,loc);
            plot(ax2,ToF,samEfd);
            xline(ax2,yPick,'b--','LineWidth',1);
            ylim(ax2,[maxSam*-1 maxSam]);
            xlim(ax2,[0 max(ToF)]);
            % axis(ax,'tight')          
        end
        
        
        function clearMemory(app)
            app.TData = [];
            app.Handler = [];
        end
        
        
        function plotSpectra(app)       
            ax = app.UIAxesFD1;
            
            clo(ax)
            clo(app.UIAxesFD2)
            axis(ax,"tight")
            cmap = app.ftColormapDropDown.Value;
            colormap(ax,cmap);
            aoiMin = app.AOIRangeFromEditField_2.Value;
            aoiMax = app.AOIRangeToEditField_2.Value;
            crange = [aoiMin aoiMax];
            clim(ax,crange);
            
            if app.ftColorbarCheckBox.Value
                colorbar(ax);
            else
                colorbar(ax,"off");                
            end

            xData = app.FData.xData;
            yData = app.FData.freq;
            magData = app.FData.magData; % magnitude data
            

            
            app.SystemStatusEditField.Value = 'Power spectra displaying...';
            drawnow;
            
            imagesc(ax,xData,yData,magData);
            ax.YDir = 'normal';

            % surf(ax,xData,yData,magData,"EdgeColor","none");
            % view(ax,2);
            app.SystemStatusEditField.Value = 'Done';
            
        end
        
                
        function plotPhases(app)

             if ~app.plotUpdate
                 return
             end
            
            ax = app.UIAxesFD2;
            app.stopProcess = 0;
            app.EnableButton.Value = false;
            
            clo(ax)
            clo(app.UIAxesFD3)
            axis(ax,"tight")
            cmap = app.ftColormapDropDown.Value;
            colormap(ax,cmap)
            
            if app.ftColorbarCheckBox.Value;
                colorbar(ax);
            else
                colorbar(ax,"off");            
            end

            xData = app.FData.xData;
            yData = app.FData.freq;
            phsData = app.FData.phsData;  % phase data
            
            app.SystemStatusEditField.Value = 'Phase displaying...';
            drawnow;
            
            imagesc(ax,xData,yData,phsData);
            ax.YDir = 'normal';
            
%             surf(ax,Data.ToF,Data.xData,real(Data.ftdSam),"EdgeColor","none");
%             view(ax,2);

            app.SystemStatusEditField.Value = 'Done';
        end
        
        
        
        
        function [data_filt, samSig_bs] = TPIDeconvolution(app, samSig)

            Data = app.TData;
            refSig = Data.Ref{2};
            blSig = Data.Ref{1};
            ToF = Data.ToF;
            filter = 'DoubleGaussian';
            HF = app.HFEditField.Value;
            LF = app.LFEditField.Value;
            ZeroPaddingFactor = app.ZeroFillingpowerofSpinner.Value;
            alpha = 1;
            
            switch filter
                case 'Wiener'
                    options = struct('beta',1000,'subbaselineref',0,...
                        'subbaselinedata',1,'debug',0);
                case 'DoubleGaussian'
                    options = struct('lf',1.5*10^-12,'hf',0.15*10^-12,...
                        'alpha',0.1,'subbaselinedata',1, ...
                        'subbaselineref',0,'debug',0);
                case 'OldImpl'
                    options = struct('cutfreq',120,...
                        'pulsewid',15,...
                        'ratio',1.5,...
                        'subbaselineref', 0,...
                        'subbaselinedata',1,'debug',0);
                otherwise 
                    error('No such filter implemented!')
            end        
                      
            N = size(samSig,2);
            
            % Set size of FFT. Find next power of two that is greater than N
            NFFT = ZeroPaddingFactor*2^nextpow2(N);
            
            if app.BLremReferenceCheckBox.Value;
                % Subtract the baseline from the reference and the raw data
                ref_data_bs = refSig - blSig;
            else
                ref_data_bs = refSig;
            end
            
            if app.BLremSampleCheckBox.Value;
                % Subtract the baseline from the reference and the raw data
                samSig_bs = samSig - blSig;
            else
                samSig_bs = samSig;
            end
            
            if N < NFFT
                size_add = (NFFT-N);
                if ~mod(size_add,2)
                    samSig_bs = [zeros(1,size_add/2);samSig_bs;zeros(1,size_add/2)];
                    ref_data_bs = [zeros(1,size_add/2);ref_data_bs;zeros(1,size_add/2)];
                else
                    samSig_bs = [zeros(1,ceil(size_add/2));samSig_bs;zeros(1,floor(size_add/2))];
                    ref_data_bs = [zeros(1,ceil(size_add/2));ref_data_bs;zeros(1,floor(size_add/2))];
                end
            end    
            
            % Transformation of raw and reference data to frequency/domain
            spec_raw = fft(samSig_bs,NFFT)/NFFT;
            spec_refSig = fft(ref_data_bs,NFFT)/NFFT;
            
            % Allocate memory for the filtered data
            data_filt = zeros(1,N);
            
            switch filter
                case 'Wiener'
                    for wave_i = 1:1
                  
                        
                        samSig_i = samSig_bs(:,wave_i);
                        spec_raw_i = spec_raw(:,wave_i);
                        
                        % Median estimator on the finest level wavelet coefficients
                        [c,l] = wavedec(mean(samSig_i),1,'db4');
                        sigma2 = wnoisest(c,l,1);
            
                        S = (norm(samSig_i-...
                            mean(samSig_i),2).^2 - N*sigma2)./...
                            norm(ref_data_bs,2).^2;
                             (abs(spec_refSig(:,wave_i)).^2 + options.beta);
                        
                        Gw = conj(spec_refSig(:,wave_i))./...
                            (abs(spec_refSig(:,wave_i)).^2 + options.beta*N*sigma2./S);
                        
                        spec_filt_wiener = spec_raw_i.*Gw;
                        
                        data_filt(:,wave_i) = real(ifft(spec_filt_wiener,NFFT))*NFFT;
            
                        % Correct the shift
                        data_filt(:,wave_i) = circshift(data_filt(1:N,wave_i),-D);
                        data_filt(:,wave_i) = data_filt(:,wave_i)./...
                            max(data_filt(:,wave_i))*max(samSig_bs(:,wave_i));
                    end   
                case 'DoubleGaussian' 
                     time_res = ToF(2) - ToF(1);
                    opticaldelay = linspace(-time_res*length(ToF)/2,time_res*length(ToF)/2,length(ToF));
            %         opticaldelay = ToF_/(3*10^8)*10^-3;
                        
                    % Get phase shift due to misplacement of reference
                    [~,idxMax] = max(ref_data_bs); 
            %         phase_shift = exp(1j*2*idxMax*opticaldelay*pi/180);
                    
                    % Define filter in time-domain
            
                    f_DG = (1/HF*exp(-ToF.^2./HF^2) - alpha*1/LF*exp(-opticaldelay.^2./LF^2));
            
                    if N < NFFT
                        size_add = (NFFT-N);
                        if ~mod(size_add,2)
                            f_DG = [zeros(1,size_add/2);f_DG;zeros(1,size_add/2)];
                        else
                            f_DG = [zeros(1,ceil(size_add/2));f_DG;zeros(1,floor(size_add/2))];
                        end
                    end
                   
                    % Transformation of filter coefficients to frequyencz domain
                    spec_filter = fft(f_DG,NFFT)./NFFT;
                            
                    spec_filt = spec_raw./spec_refSig.*spec_filter;
                    
                    % Correct for phase shift
                    ph = unwrap(angle(1./spec_refSig.*spec_filter));
                    spec_filt = spec_filt.*exp(-1j*ph);
                   
                    % Transformation to time-domain
                    data_filt = real(ifft(spec_filt,NFFT))*NFFT;
                    
                    % Truncate the filtered data (due to zero padding)
                    if N < NFFT 
                        size_add = (NFFT-N); 
                        if ~mod(size_add,2)
                            data_filt = data_filt(size_add/2+1:N+size_add/2)./...
                                repmat(max(data_filt),1,N).*repmat(max(samSig_bs),1,N);    
                        else
                            data_filt = data_filt(floor(size_add/2)+1:N+ceil(size_add/2))./...
                                repmat(max(data_filt),1,N).*repmat(max(samSig_bs),1,N);    
                        end
                    else
                        data_filt = data_filt(1:N)./...
                            repmat(max(data_filt),1,N).*repmat(max(samSig_bs),1,N);
                    end    
                    
                case 'OldImpl'     
                    data_filt = myDeconvolutionNew(ref_data_bs(1,:),...
                            samSig_bs,...
                            ToF,...
                            options.cutfreq,...
                            options.pulsewid,...
                            options.ratio);
                end
            end
            
        
        function posT2_YLine(app)
            %yPick = app.tPickSlider_Y.Value;
            xPick = app.LocationSlider.Value;
            %sf = 1/app.ToFSpacingpsEditField.Value;
            ax3 = app.UIAxesFD3;
            ax4 = app.UIAxesFD4;
            
            try 
                xData = app.FData.xData;
                freq = app.FData.freq;
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'No filtered data set','Warning');
                app.EnableButton_2.Value = 0;
                EnableButton_2ValueChanged(app,0);
                app.SystemStatusEditField.Value = 'Enabling cancelled';
                return;
            end
            xData = app.FData.xData;
            freq = app.FData.freq;

            cla(ax3)
            cla(ax4)
            
            grid(ax3,"on")
            grid(ax4,"on")
            app.Handler.xline_21.Value = xPick;
            app.Handler.xline_22.Value = xPick;        
                    
            % plot THz signal of the cross pointed location
            xLoc = sum(xData<=xPick);
            app.FData.xPick = xLoc;
            magVec = magData(xLoc,:);
            phsVec = phaData(xLoc,:);

            yLim =[min(ftdData,[],'all'), max(ftdData,[],'all')];
            sigAmp = ftdData(:,dLoc);
            
            %title(ax3,'E field Change');
            plot(ax3,xData,sigAmp);
            plot(ax4,ingressDepth,ftdVec);
            title(ax4,'Single Filtered Waveform');
            
            % axis(ax3,'tight') 
            % axis(ax4,'tight')
            app.Handler.xline_23 = xline(ax3,xData(xLoc) ,'r--','LineWidth',1);
            app.Handler.xline_24 = xline(ax4,xPick,'r--','LineWidth',1);
            
            ax4.YLim = yLim;
            ax3.YLim = yLim;
        end
        
        function [freq outSig] = powerspec(app,inSig,sf)
            n = length(inSig);
            inSigH = fft(inSig,n);
            PS = inSigH.*conj(inSigH)/n;
            freq = sf*(0:n)/n;
            L = 1:floor(n/2);
            freq = freq(L);
            outSig = PS(L);    
        end
        
        function posT2_XLine(app)
            xPick = app.LocationSlider.Value;
            xData = app.FData.xData;
            freq = app.FData.freq;
            magData = app.FData.magData;
            phsData = app.FData.phsData;

            ax3 = app.UIAxesFD3;
            ax4 = app.UIAxesFD4;

            app.Handler.xline_21.Value = xPick;
            app.Handler.xline_22.Value = xPick;
            
            cla(ax3)
            cla(ax4)
            
            grid(ax3,"on")
            grid(ax4,"on")
                    
            % plot THz signal of the cross pointed location
            xLoc = sum(xData<=xPick);
            app.FData.xPick = xLoc;
            magVec = magData(:,xLoc);
            phsVec = phsData(:,xLoc);

            [amin amax] = bounds(magData,"all");
            [pmin pmax] = bounds(phsData,"all");
            
            %title(ax3,'E field Change');
            plot(ax3,freq,magVec);
            plot(ax4,freq,phsVec);
            
            % axis(ax3,'tight') 
            % axis(ax4,'tight')
            
            ax3.YLim = [amin amax];
            ax4.YLim = [pmin pmax];
        end
        
        function LFPlot(app)
            samData = app.TData.samData;
            xData = app.TData.xData;
            displacement = app.TData.displacement;
            alp = str2num(app.AlphaDropDown.Value);
            cmap = app.ExtColormapDropDown.Value;
            thickness = app.ThicknessmmEditField_LE.Value;
            upto3mm = app.Upto3mmCheckBox.Value;
            normOpt = 0; %normalise two sets of data
            
            if normOpt
                samData = samData/max(samData,[],'all');
            end
           
            ax1 = app.UIAxesLE1;
            ax2 = app.UIAxesLE2;
            ax3 = app.UIAxesLE3;
            
            cla(ax1);
            cla(ax2);
            cla(ax3);
            
            imagesc(ax1,xData,displacement,samData,'AlphaData',alp);
            axis(ax1,"xy");            
            axis(ax1,'tight');

            if upto3mm
                ylim(ax1,[0, 3]);
                ylim(ax3,[0, 3]);
            end

            if app.DisplayTabletCentreCheckBox.Value
                hold(ax1,"on");
                yline(ax1,thickness/2,'--','Tablet Centre');
            end
            
            colormap(ax1,cmap);

            app.LiquidIngressTimesecEditField.Value = 0;         
        end
        
        
        function updateBatchList(app)
            % batcht list update
            bNum = size(app.CData.batch,2);
            ListBoxItems={};
            cnt = 1;
            
            for bIdx = 1:bNum
                AddItem = app.CData.batch{bIdx};
                if ~isempty(AddItem)&&~sum(strcmp(AddItem,ListBoxItems))
                   ListBoxItems(cnt) = {AddItem};
                   cnt = cnt+1;
                end
            end
            
            app.BatchListBox.Items = ListBoxItems;
            app.BatchListBox.ItemsData = (1:length(ListBoxItems));           
        end
        
        function CDPlot_Individual(app)
            itemNums = app.MeasurementListBox.Value;
            itemNames = app.MeasurementListBox.Items;
            
            if isempty(itemNums)
                fig = app.DipTabUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            Peaks = app.CData.Peaks;
            
            ax1 = app.UIAxesCD1;
            ax2 = app.UIAxesCD2;
            axis(ax1,'tight');
            axis(ax2,'tight');
            xlim(ax1,[0 inf]);
            ylim(ax1,[0 inf]);
            cla(ax1);
            cla(ax2);
            hold(ax1,'on');   
            hold(ax2,'on');         
            
            lh1 = [];
            lh2 = [];
            cnt = 1;
            
            for idx = itemNums
                xLocs = Peaks{idx}.xLocs;    
                eFieldAmp = Peaks{idx}.eFieldAmp;
                yLocs = Peaks{idx}.yLocs;

                h1 = plot(ax1,xLocs,yLocs,'.');
                h2 = plot(ax2,xLocs,eFieldAmp,'.');
                
                lh1(cnt) = h1;
                lh2(cnt) = h2;
                
                cnt = cnt + 1;
            end
            
            if app.LegendCheckBox.Value
                legend(ax1,lh1,itemNames([itemNums]),"Location","southeast","Interpreter","none");
                legend(ax2,lh2,itemNames([itemNums]),"Location","southeast","Interpreter","none");                
            else
                legend(ax1,"off");
                legend(ax2,"off");
            end
        end
        
        function CDPlot_Batch(app)
            bItems = app.BatchListBox.Value;
            sBatch = app.BatchListBox.Items(bItems); % sBatch: selected batch(es)
             
            if isempty(bItems)
                fig = app.DipTabUIFigure;
                uialert(fig,'Select batch(es)','Warning');
                return;
            end
            
            Peaks = app.CData.Peaks;
            batch = app.CData.batch;
            meta = app.CData.meta;
            totMeasNum = size(Peaks,2); % total measurement number
            bMat = {}; % batch data matrix
            
            ax1 = app.UIAxesCD1;
            ax2 = app.UIAxesCD2;
            axis(ax1,'tight');
            axis(ax2,'tight');
            xlim(ax1,[0 inf]);
            ylim(ax1,[0 inf]);
            cla(ax1);
            cla(ax2);
            hold(ax1,'on');
            hold(ax2,'on');            
            
            lh1 = []; % legend list 1
            lh2 = []; % legend list 2
            cnt = 1;
            
            for bIdx = sBatch 
                bIdxMat = strcmp(batch,bIdx); % returns measurement lists that match with bIdx in a vector form 
                bMat(cnt,1:2) = {cnt,bIdx}; % batch data cell structure
                bNum = sum(bIdxMat); % total number of nonzero (identical batch name) items
                batchXLocs = [];
                batchYLocs = [];
                batchEFieldAmp = [];
                
                for idx = 1:totMeasNum
                    if bIdxMat(idx) % read only a selected batch measurement
                        batchXLocs =  [batchXLocs, Peaks{idx}.xLocs];
                        batchYLocs =  [batchYLocs, Peaks{idx}.yLocs];
                        batchEFieldAmp = [batchEFieldAmp, Peaks{idx}.eFieldAmp];
                    end
                end

                bMat(cnt,3) = {[batchXLocs; batchYLocs]};
                bMat(cnt,4) = {[batchXLocs; batchEFieldAmp]};

                cnt = cnt + 1;
            end

            app.CData.bMat = bMat;

            cnt2 = 1;
            
            for bIdx = sBatch
                pkPints = bMat{cnt2,3}; % peak points
                pkAmps = bMat{cnt2,4}; % peak amplitudes
                p1 = plot(ax1,pkPints(1,:),pkPints(2,:),'.');
                p2 = plot(ax2,pkAmps(1,:),pkAmps(2,:),'.');
                lh1(cnt2) = p1;
                lh1(cnt2) = p2;
                cnt2 = cnt2 + 1;
            end
                                    
            if app.LegendCheckBox.Value
                legend(ax1,lh1,sBatch,"Location","southeast","Interpreter","none");
                legend(ax2,lh2,sBatch,"Location","southeast","Interpreter","none");
            else
                legend(ax1,"off");
                legend(ax2,"off");
            end
         
        end
      
        
        
        function output = TDSunwrap(app,pData,freq)
            % unwrapping functon
            %freq = app.FD_data.frequency{idx};
            strFreq = app.StartFrequencyTHzEditField.Value;
            
            %unwrapping starting frequency: 0.8THz due to its high SNR
            strFreq = strFreq * 1e12;
            % find the index of the first element in 'freq' that has a value
            % greater than 'srtFreq'
            srtLoc = find(freq > strFreq,1);
            % using srtLoc as a starting point, and in the order of increasing indices, unwrap phase values to the
            % end of the data
            pData1 = unwrap(pData(srtLoc:end));
            % using srtLoc as a starting point, and in the order of decreasing indices, unwrap phase values to the
            % start of the data
            % this action reorders the data from back to front in pData2
            pData2 = unwrap(pData(srtLoc:-1:1));
            % reordering phase values, to from front to back, excluding the
            % phase value at index strLoc, since its phase is already
            % included in pData1
            pData3 = pData2(end:-1:2);
            % grouping all phase values into one single vector
            pData = [pData3 pData1];
            
            % default: extrapolate phase data from 0.2 to 0.4 THz to estimate the
            % shift in phase and correct for it
            % more noise seems to be resulted if the lower limit of the specified region 
            % takes a value less than 0.2 THz
            epol_srtFreq = app.FromEpolFreqEditField.Value * 1e12;
            epol_endFreq = app.ToEpolFreqEditField.Value * 1e12;
            % find the index of the first element in 'freq' that has a value
            % greater than 'epol_srtFreq'
            epol_srtLoc = find(freq > epol_srtFreq,1);
            % find the index of the first element in 'freq' that has a value
            % greater than 'epol_endFreq'
            epol_endLoc = find(freq > epol_endFreq,1);
            % extracting frequency values from 'epol_srtFreq' to 'epol_endFreq'
            epol_freq = freq(epol_srtLoc:epol_endLoc);
            % extracting phase data that lies in frequency values from 'epol_srtFreq' to 'epol_endFreq'
            epol_data = pData(epol_srtLoc:epol_endLoc);
            
            % fitting a straight line of extracted phase against extracted
            % frequency
            p = polyfit(epol_freq,epol_data,1);
            % find the intercept of the fitted line
            shift = p(2); % y-axis intersection point value
            % shift all phase data down by the intercept value
            output = pData - shift;
        end
        
        function updateTable(app)
            Tcell = app.Tcell;
            measNum = size(Tcell,2);

            sFont = uistyle("FontColor","black");
            addStyle(app.UITable,sFont);
            app.UITable.Data = cell2table(Tcell);
        end
    end


    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            
        end

        % Button pushed function: ImportthzFileButton
        function ImportthzFileButtonPushed(app, event)
            
            [filename, pathname] = uigetfile('*.thz');

            if isequal(filename,0)||isequal(pathname,0)
                return;          
            end
            
            fileinfo = strcat(pathname,filename);
            app.filefullpath = fileinfo;
                       
            ProjectName = strrep(filename,'.thz','');
            app.ProjectNameEditField.Value = ProjectName;
            
            app.DeployButton.Enable = true;
            figure(app.DipTabUIFigure);            
        end

        % Button pushed function: LoadProjectButton
        function LoadProjectButtonPushed(app, event)
            [filename, filepath] = uigetfile('*.mat');
            
            if isequal(filename,0)||isequal(filepath,0)
                return;
            end
            
            fullfile = strcat(filepath,filename);
            app.SystemStatusEditField.Value = 'Project loading...';
            drawnow
            
            %clearMemory(app);
            load(fullfile);
            app.CData = CData;
            app.Tcell = Tcell;
            app.MeasurementListBox.Items = ItemList;
            app.MeasurementListBox.ItemsData = (1:length(ItemList));

            updateTable(app);            
            updateBatchList(app);
            app.SystemStatusEditField.Value = 'Project loaded.';
        end

        % Button pushed function: PlotButton
        function PlotButtonPushed(app, event)
            TDPlot(app);        
        end

        % Value changed function: ColormapDropDown
        function ColormapDropDownValueChanged(app, event)
            value = app.ColormapDropDown.Value;
            TDPlot(app);
        end

        % Value changed function: EnableButton
        function EnableButtonValueChanged(app, event)
            value = app.EnableButton.Value;
            ax1 = app.UIAxesTD1;
            ax2 = app.UIAxesTD2;
            ax3 = app.UIAxesTD3;
            
            try 
                xData = app.TData.xData;
                ToF = app.TData.ToF;
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'Load Data first','Warning');
                app.EnableButton.Value = false;
                return;
            end
            
            
            tXRange = double(([min(xData) max(xData)]));
            tYRange = double(round([min(ToF) max(ToF)]));
            
            if value
                %Enable Buttons
                app.SpectrogramButton.Enable = true;
                app.PowerSpectrumButton.Enable = true;
                
                app.tXPickSlider.Enable = true;
                app.tXPickEditField.Enable = true;
                app.tYPickSlider.Enable = true;
                app.tYPickEditField.Enable = true;
                app.SetDownLimitButton.Enable = true;
                app.SetLeftLimitButton.Enable = true;
                
                % limits settting
                app.tXPickSlider.Limits = tXRange;
                app.tXPickEditField.Limits = tXRange;
                initX = mean(tXRange);

                app.tYPickSlider.Limits = tYRange;
                app.tYPickEditField.Limits = tYRange;
                initY = mean(tYRange);
                
                % set current position values                
                app.tXPickSlider.Value = initX;
                app.tXPickEditField.Value = initX;
                app.tYPickSlider.Value = initY;
                app.tYPickEditField.Value = initY;
                
                % define x,y line handler
                app.Handler.xline = xline(ax1,initX,'R--','LineWidth',1);
                app.Handler.yline = yline(ax1,initY,'B--','LineWidth',1);
                % cummurative E-field indicator line
                app.Handler.CExline = xline(ax3,initX,'R--','LineWidth',1);
                posT1_XYLine(app);
            else
                app.SpectrogramButton.Enable = false;
                app.PowerSpectrumButton.Enable = false;
                app.tXPickSlider.Enable = false;
                app.tXPickEditField.Enable = false;
                app.tYPickSlider.Enable = false;
                app.tYPickEditField.Enable = false;
                app.SetDownLimitButton.Enable = false;
                app.SetLeftLimitButton.Enable = false;
                TDPlot(app);
            end
            
        end

        % Value changed function: tXPickSlider
        function tXPickSliderValueChanged(app, event)
            value = app.tXPickSlider.Value;
            app.tXPickEditField.Value = value;
            
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Button pushed function: StopProcessButton
        function StopProcessButtonPushed(app, event)
            app.stopProcess = true;
        end

        % Button pushed function: SaveFigureButton
        function SaveFigureButtonPushed(app, event)
            ax = app.UIAxesTD1;
            filter = {'*.pdf';'*.*'};
            [filename, filepath] = uiputfile(filter);
            
            if isequal(filename,0)||isequal(filepath,0)
                return;          
            end
            
            fullfile = strcat(filepath,filename);
            exportgraphics(ax,fullfile);
        end

        % Value changed function: tXPickEditField
        function tXPickEditFieldValueChanged(app, event)
            value = app.tXPickEditField.Value;
            app.tXPickSlider.Value = value;
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Button pushed function: TruncateButton
        function TruncateButtonPushed(app, event)
            samData = app.TData.samData;
            rawData = app.TData.rawData;
            xData = app.TData.xData; % x-axis
            ToF = app.TData.ToF; % time of flight (ps)
            
            app.DeployButton.Enable = false; % prevent performing twice
            app.SystemStatusEditField.Value = 'Truncating....';

            if app.EnableButton.Value
                app.EnableButton.Value = false;
                EnableButtonValueChanged(app, event)
            end

            drawnow
                                              
            dataLength=size(samData,1);
            measNum = size(samData,2);
            
            %Y,X-trancation option
            xlbLoc = 1;
            xubLoc = measNum;
            ylbLoc = 1;
            yubLoc = dataLength;
            
            if app.LeftEditField.Value
                xlb = app.LeftEditField.Value;
                xlbLoc = sum(xData <= xlb);              
            end
            
            if app.RightEditField.Value
                xub = app.RightEditField.Value;
                xubLoc = sum(xData <= xub);
            end

            if app.UpEditField.Value
                ylb = app.UpEditField.Value;
                ylbLoc = sum(ToF <= ylb);              
            end
            
            if app.DownEditField.Value
                yub = app.DownEditField.Value;
                yubLoc = sum(ToF <= yub);
            end
            
            % lb, ub availability check
            if xlbLoc==0 || xubLoc>measNum || xlbLoc>xubLoc
                fig = app.DipTabUIFigure;
                uialert(fig,'Incorrect X-truncation setting','Warning');
                app.DeployButton.Enable = true;
                app.TruncateButton.Enable = true;
                app.SystemStatusEditField.Value = 'NEXT cancelled';
                return;
            end

           % lb, ub availability check
            if ylbLoc==0 || yubLoc>dataLength || ylbLoc>yubLoc
                fig = app.DipTabUIFigure;
                uialert(fig,'Incorrect Y-truncation setting','Warning');
                app.DeployButton.Enable = true;
                app.TruncateButton.Enable = true;
                app.SystemStatusEditField.Value = 'NEXT cancelled';
                return;
            end

            samData = samData(ylbLoc:yubLoc, xlbLoc:xubLoc);
            rawData = rawData(ylbLoc:yubLoc, xlbLoc:xubLoc);
            
            [cmin cmax] = bounds(samData,"all");

            cmin = round(cmin*10^2)*10^-2;
            cmax = round(cmax*10^2)*10^-2;

            app.DataRangeFromEditField.Value = cmin;
            app.DataRangeToEditField.Value = cmax;
            app.AOIRangeFromEditField.Value = cmin;
            app.AOIRangeToEditField.Value = cmax;
                        
            % Scan information panel display
            app.dataLengthEditField_FD.Value = dataLength;
            
            % assign truncated Y,X time
            xData = xData(xlbLoc:xubLoc) - xData(xlbLoc);
            app.TData.xData = xData;
            ToF = ToF(ylbLoc:yubLoc) - ToF(ylbLoc);
            app.TData.ToF = ToF;
            
            app.SystemStatusEditField.Value = 'Done';
            app.TData.samData = samData;
            app.TData.rawData = rawData;
            % app.TData.ftdSam = samData;
            app.DeployButton.Enable = true;
            app.LeftEditField.Value = 0;
            app.RightEditField.Value = 0;
            app.UpEditField.Value = 0;
            app.DownEditField.Value = 0;
            drawnow
            TDPlot(app);
        end

        % Button pushed function: LfPlotButton
        function LfPlotButtonPushed(app, event)
            LFPlot(app);
            app.kmmsEditField.Value = 0;
            app.dmmEditField.Value = 0;
            app.R2EditField.Value = 0;
            app.RMSEEditField.Value = 0;
            app.SystemStatusEditField.Value = 'Replot Finished';
        end

        % Button pushed function: SaveProjectButton
        function SaveProjectButtonPushed(app, event)
            filter = {'*.mat';'*.*'};
            [filename, filepath] = uiputfile(filter);
            
            if isequal(filename,0)||isequal(filepath,0)
                return;          
            end
            
            fullfile = strcat(filepath,filename);
            app.SystemStatusEditField.Value = 'Project saving...';
            drawnow
            
            CData = app.CData;
            Tcell = app.Tcell;
            ItemList = app.MeasurementListBox.Items;            
            
            save(fullfile,'CData','Tcell','ItemList');
            app.SystemStatusEditField.Value = 'Project saved.';
        end

        % Button pushed function: SpectrogramButton
        function SpectrogramButtonPushed(app, event)
            app.stopProcess = 0;
            app.SystemStatusEditField.Value = "Spectrogram";
            loc = app.TData.xPick;
            sf = 1/app.ToFSpacingpsEditField.Value;
            samSig = app.TData.samData(:,loc);
            ToF = app.TData.ToF;
            xData = app.TData.xData;
            ax = app.UIAxesTD4;

            cla(ax)
            axis(ax,'tight');
            title(ax,'Spectrogram');
            xlabel(ax,'Time (ps)');
            ylabel(ax,'Frequency (THz)');
            
            if app.AutoScanSpectrumCheckBox.Value
                for idx = 1:size(app.TData.samData,2)
                    app.tXPickSlider.Value = xData(idx);
                    app.tXPickSlider.ValueChangedFcn(app, event);
                    samSig = app.TData.samData(:,idx);
                    [S,F,T] = pspectrum(samSig,sf,'spectrogram','FrequencyLimits',[0.8,1.1],'MinThreshold',-110);
                    imagesc(ax,ToF,flipud(F),rot90(log(abs(S'))));
                    set(ax,'YDir','normal');
                    
                    if app.stopProcess
                        app.SystemStatusEditField.Value = "Process aborted";
                        app.SpectrogramButton.Enable = true;
                        return
                    end
                    
                    drawnow
                end
            else
                [S,F,T] = pspectrum(samSig,sf,'spectrogram','FrequencyLimits',[0.8,1.1],'MinThreshold',-110);
                imagesc(ax,ToF,flipud(F),rot90(log(abs(S'))));
                set(ax,'YDir','normal');
                drawnow            
            end
            
            axis(ax,'tight');     
        end

        % Value changing function: tXPickSlider
        function tXPickSliderValueChanging(app, event)
            changingValue = event.Value;
            app.tXPickEditField.Value = changingValue;
            
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Button pushed function: PowerSpectrumButton
        function PowerSpectrumButtonPushed(app, event)
            app.stopProcess = 0;
            app.SystemStatusEditField.Value = "Power Spectrum";
            try
                loc = app.TData.xPick;
            catch
                return;
            end

            sf = 1/app.ToFSpacingpsEditField.Value;
            samSig = app.TData.samData(:,loc);
            ToF = app.TData.ToF;
            xData = app.TData.xData;
            ax4 = app.UIAxesTD4;
            yULim = 0; % y-axis upper limit

            cla(ax4)
            axis(ax4,'tight');
            grid(ax4,"on")
            title(ax4,'Power Spectrum');
            xlabel(ax4,'Frequency (THz)');
            ylabel(ax4,'E filed (a.u.)');
            
            if app.AutoScanSpectrumCheckBox.Value
                for idx = 1:size(app.TData.samData,2)                    
                    app.tXPickSlider.Value = xData(idx);
                    app.tXPickSlider.ValueChangedFcn(app, event);
                    samSig = app.TData.samData(:,idx);
                    
                    n = length(samSig);
                    samSigH = fft(samSig,n);
                    PS = samSigH.*conj(samSigH)/n;
                    freq = sf*(0:n)/n;
                    L = 1:floor(n/2);

                    yMax = max(PS(sum(freq<0.2):end));
                    if yULim <= yMax;
                        yULim = yMax;
                    end
  
                    plot(ax4,freq(L),PS(L));
                    ax4.YLim = [0 yULim];
                    ax4.XLim = [0 3];
                    
                    if app.stopProcess
                        app.SystemStatusEditField.Value = "Process aborted";
                        app.SpectrogramButton.Enable = true;
                        return
                    end
                    
                    drawnow
                end
            else
                [freq PS] = powerspec(app,samSig,sf);

                plot(ax4,freq,PS);
                % ax.XLim = [0 3];
                % ax.YLim = [0 max(PS(sum(freq<0.2):end))*1.1];
                ax4.XLim = [0 5];
                ax4.YLim = [0 10];
                drawnow            
            end
            
%             axis(ax,'tight');     
        end

        % Value changed function: EnableButton_2
        function EnableButton_2ValueChanged(app, event)
            value = app.EnableButton_2.Value;
            ax1 = app.UIAxesFD1;
            ax2 = app.UIAxesFD2;
            ax3 = app.UIAxesFD3;
            ax4 = app.UIAxesFD4;
            
            try 
                xData = app.FData.xData;
                freq = app.FData.freq;
                magData = app.FData.magData;
                phsData = app.FData.phsData;
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'Load Data first','Warning');
                app.EnableButton.Value = false;
                return;
            end
            
            xRange = double(([min(xData) max(xData)]));
            
            if value
                %Enable Buttons
                app.LocationSlider.Enable = true;
                app.LocationSlider.Limits = xRange;
                xCur = mean(xRange);
                app.LocationSlider.Value = xCur;

                app.AutoScanButton.Enable = true;
                app.NextButton.Enable = true;
              
                % define x,y line handler
                app.Handler.xline_21 = xline(ax1,xCur,'w--','LineWidth',1);
                app.Handler.xline_22 = xline(ax2,xCur,'w--','LineWidth',1);
                posT2_XLine(app);
            else
                app.LocationSlider.Enable = false;
                app.AutoScanButton.Enable = false;
                app.NextButton.Enable = false;
                
                try 
                    app.Handler.xline_21.Visible = false;
                    app.Handler.xline_22.Visible = false;
                catch ME
                    
                end
        
            end
        end

        % Button pushed function: AutoScanButton
        function AutoScanButtonPushed(app, event)
            app.stopProcess = 0;
            app.SystemStatusEditField.Value = "Auto Scan in Process";
            magData = app.FData.magData;
            phsData = app.FData.phsData;
            xData = app.FData.xData;
            freq = app.FData.freq;
            app.AutoScanButton.Enable = false;
            
            for idx = 1:length(xData)
                app.LocationSlider.Value = xData(idx);
                app.LocationSlider.ValueChangedFcn(app, event);
                posT2_XLine(app);
                
                if app.stopProcess
                    app.SystemStatusEditField.Value = "Process aborted";
                    app.AutoScanButton.Enable = true;
                    return
                end
                
                drawnow
            end
            
            app.AutoScanButton.Enable = true;
        end

        % Button pushed function: TransformButton
        function TransformButtonPushed(app, event)
            app.stopProcess = 0;
            app.SystemStatusEditField.Value = "FFT in Process";
            
            if app.EnableButton_2.Value
                app.EnableButton_2.Value = 0;
                EnableButton_2ValueChanged(app,0); 
            end
            
            samData = app.TData.samData;
            measNum = size(samData,2);
            %ToF = app.TData.ToF;
            xData = app.TData.xData;

            fs = 1/(app.ToFSpacingpsEditField.Value*10^-12);
            min_freq = app.FromFreqEditField.Value;
            max_freq = app.ToFreqEditField.Value;
            upscale = app.ZeroFillingpowerofSpinner.Value; 
            funcName = app.FunctionDropDown.Value; %window function
           
            for idx = 1:measNum
                TD_sample = samData(:,idx);
                n = length(TD_sample);

                wf = str2func(funcName);
                TD_sample = TD_sample.*window(wf,length(TD_sample))';
                N = 2^(nextpow2(length(TD_sample))+upscale);
                FD_sample = fft(TD_sample,N);
                freqs = 0:fs/N:fs/2;
                % freqRes = fs/(samNum*10^12);
                % freqRes_padded = fs/(N*10^12);
                cutoff_low = sum(freqs < min_freq*10^12) + 1;
                cutoff_high = sum(freqs < max_freq*10^12);
                
                FD_frequency = freqs(1:cutoff_high);                
                FD_sample = FD_sample(1:cutoff_high);

                % unwrap phase data (calling TDSunwrap function)
                % unwrapping phase values for reference and sample
                % measurements in frequency domain
                % the negative sign is added to compensate for the negative sign introduced during the fast Fourier transform process
                uw_samPhase = TDSunwrap(app,-angle(FD_sample),FD_frequency);
                
                % cut off lower frequency part
                % trim lower end values using the lower cutoff value
                % lower end values can only be trimmed now since trimming
                % them earlier would affect the phase values obtained
                % from unwrapping
                FD_frequency = FD_frequency(cutoff_low:end);
                FD_sample = FD_sample(cutoff_low:end);
                uw_samPhase = uw_samPhase(cutoff_low:end);
                
                % allocating FData
                ffdData{idx} = FD_sample';
                sam_magnitude{idx} = abs(FD_sample)';
                sam_phase{idx} = uw_samPhase';

                progressP = idx/measNum*100;
                progressP = num2str(progressP,'%.0f');
                progressP = strcat("FFT: ", progressP,"%");
                app.SystemStatusEditField.Value = progressP;
                
                if app.stopProcess
                    app.SystemStatusEditField.Value = "Process aborted";
                    app.SpectrogramButton.Enable = true;
                    return
                end
                
                drawnow
            end

            app.FData.xData = xData;
            app.FData.freq = FD_frequency*10^-12;
            app.FData.cpxData = cell2mat(ffdData);
            app.FData.magData = cell2mat(sam_magnitude);
            app.FData.phsData = cell2mat(sam_phase);

            [cmin cmax] = bounds(app.FData.magData,"all");

            cmin = round(cmin*10^2)*10^-2;
            cmax = round(cmax*10^2)*10^-2;

            app.DataRangeFromEditField_2.Value = cmin;
            app.DataRangeToEditField_2.Value = cmax;
            app.AOIRangeFromEditField_2.Value = cmin;
            app.AOIRangeToEditField_2.Value = cmax;

            % [xmin xmax] = bounds(xData,"all");
            % app.LocationSlider.Limits = [xmin xmax];
            % app.LocationSlider.Value = (xmin+xmax)/2;
            app.EnableButton_2.Value  = false;
            app.EnableButton_2ValueChanged;

            %Display
            plotSpectra(app);
            plotPhases(app);
        end

        % Button pushed function: NextButton
        function NextButtonPushed(app, event)
            try
                rawData = app.TData.rawData;
                samData = app.TData.samData;
                xData = app.TData.xData;
                ToF = app.TData.ToF;
                app.TData.algoROI = [];
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                return;
            end

            LFPlot(app);
          
            app.TabGroup.SelectedTab = app.TabGroup.Children(3);       
        end

        % Button pushed function: DataComparisonButton
        function DataComparisonButtonPushed(app, event)
            try
                Peaks = app.TData.Peaks;
                xData = app.TData.xData;
                displacement = app.TData.displacement;
                ingressTime = app.LiquidIngressTimesecEditField.Value;
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                return;
            end
                        
            meta.sampleName = app.SampleNameEditField.Value;
            meta.thickness = app.ThicknessmmEditField_LE.Value;
            meta.refractiveIndex = app.RefractiveIndexEditField_LE.Value;
            meta.ingressTime = ingressTime;
            parameter.k = app.kmmsEditField.Value;
            parameter.d = app.dmmEditField.Value;
            parameter.R2 = app.R2EditField.Value;
            parameter.RM = app.RMSEEditField.Value;
            batch = {''};
 
            try
                measNum = size(app.CData.Peaks,2)+1;
            catch ME
                app.CData.Peaks = {};
                measNum = 1;
            end

            if app.AutoSaveFiguresCheckBox.Value
               SaveFigureButton_LEPushed(app);
            end
            
            app.CData.Peaks(measNum) = {Peaks};
            app.CData.meta(measNum) = meta;
            app.CData.batch(measNum) = batch;
            app.CData.parameter(measNum) = {parameter};

            % Table cell update
            app.Tcell{measNum,1} = meta.sampleName;
            app.Tcell{measNum,2} = meta.thickness;
            app.Tcell{measNum,3} = meta.refractiveIndex;
            app.Tcell{measNum,4} = meta.ingressTime;
            app.Tcell{measNum,5} = parameter.k;
            app.Tcell{measNum,6} = parameter.d;
            app.Tcell{measNum,7} = parameter.R2;
            app.Tcell{measNum,8} = parameter.RM;            
            
            % Measurment list update
            ListBoxItems={};
            
            for MeasNum = 1:measNum
                AddItem = app.CData.meta(MeasNum).sampleName;
                ListBoxItems{MeasNum} = AddItem;
            end
            
            app.MeasurementListBox.ItemsData = (1:MeasNum);
            app.MeasurementListBox.Items = ListBoxItems;
            updateTable(app);
            
            app.TabGroup.SelectedTab = app.TabGroup.Children(3);           
        end

        % Button pushed function: RemoveButton_DC1
        function RemoveButton_DC1Pushed(app, event)
            delItem = app.MeasurementListBox.Value;
            ListBoxItems = app.MeasurementListBox.Items;
            
            if isempty(delItem)
                return;
            end
            
            app.CData.Peaks(delItem) = [];
            app.CData.meta(delItem) = [];
            app.CData.batch(delItem) = [];
            ListBoxItems(delItem) = [];
            app.Tcell(delItem,:) = [];

            app.MeasurementListBox.Items = ListBoxItems;
            app.MeasurementListBox.ItemsData = (1:length(ListBoxItems));
            updateTable(app);
            
            MeasurementListBoxValueChanged(app);
        end

        % Button pushed function: AssigndatainworkspaceButton
        function AssigndatainworkspaceButtonPushed(app, event)
              assignin('base',"CData",app.CData);
              assignin('base',"Tcell",app.Tcell);
        end

        % Value changed function: MeasurementListBox
        function MeasurementListBoxValueChanged(app, event)
            value = app.MeasurementListBox.Value;
            
        end

        % Button pushed function: GroupButton
        function GroupButtonPushed(app, event)
            bItem = app.MeasurementListBox.Value;          
            bName = {app.BatchNameEditField.Value};
            
            if isempty(bItem)||isequal(bName,'')
                return;
            end
            
            app.CData.batch(bItem) = bName;           
            updateBatchList(app);
            
        end

        % Value changed function: BatchListBox
        function BatchListBoxValueChanged(app, event)
            value = app.BatchListBox.Value;
            sBatch = app.BatchListBox.Items(value);            
            tNum = size(app.CData.batch,2); % data set number
            ListBoxItems={};
            cnt = 1;
            
            for idx = 1:tNum
                meta = app.CData.meta(idx);
                AddItem = {meta.sampleName};
                cBatch = app.CData.batch(idx);
                if ~isempty(AddItem)&&sum(strcmp(cBatch,sBatch))
                   ListBoxItems(cnt) = AddItem;
                   cnt = cnt+1;
                end
            end
            
            app.BatchDetailListBox.Items = ListBoxItems;
            app.BatchDetailListBox.ItemsData = (1:length(ListBoxItems));   
            
        end

        % Button pushed function: UngroupButton
        function UngroupButtonPushed(app, event)
            delItem = app.BatchDetailListBox.ItemsData;
            ListBoxItems = app.BatchDetailListBox.Items;
            
            if isempty(delItem)
                return;
            end
            
            app.CData.batch(delItem) = {''};
            ListBoxItems(delItem) = [];

            app.BatchDetailListBox.Items = ListBoxItems;
            app.BatchDetailListBox.ItemsData = (1:length(ListBoxItems));
                    
            updateBatchList(app);
        end

        % Button pushed function: AddButton
        function AddButtonPushed(app, event)
            tBatch = app.BatchListBox.Value; %target batch
            bItem = app.MeasurementListBox.Value; %selected measurements         
            
            if isequal(size(tBatch,2),1)
                bName = app.BatchListBox.Items(tBatch);
                app.CData.batch(bItem) = bName;
            else
                return;
            end
            
            MeasurementListBoxValueChanged(app);
            BatchListBoxValueChanged(app);
            updateBatchList(app);
            
        end

        % Button pushed function: RemoveButton_DC2
        function RemoveButton_DC2Pushed(app, event)
            delItem = app.BatchDetailListBox.Value;
            ListBoxItems = app.BatchDetailListBox.Items;
            
            if isempty(delItem)
                return;
            end
            
            tNum = size(app.CData.batch,2); % data set number
            
            for idx = 1:tNum
                meta = app.CData.meta(idx);
                cMeas = {meta.description};
                sMeas = ListBoxItems(delItem);
                if sum(strcmp(cMeas,sMeas))
                   app.CData.batch{idx}='';
                end
            end
            
            MeasurementListBoxValueChanged(app);
            BatchListBoxValueChanged(app);
           
        end

        % Button pushed function: PlotButton_DC
        function PlotButton_DCPushed(app, event)
            plotTarget = app.PlotButtonGroup.SelectedObject.Text;
            
            if isequal(plotTarget,'Individual')
                CDPlot_Individual(app);
            else
                CDPlot_Batch(app);
            end
        end

        % Button pushed function: CaculateFittingParametersButton
        function CaculateFittingParametersButtonPushed(app, event)
            try
                Peaks = app.TData.Peaks;
                xData = app.TData.xData;
                thickness = app.ThicknessmmEditField_LE.Value;
            catch
                fig = app.DipTabUIFigure;
                uialert(fig,'Invalid Datasets','Warning');
                return;
            end

            xLocs = Peaks.xLocs;
            yLocs = Peaks.yLocs;
            
            ax1 = app.UIAxesLE1;
            ax3 = app.UIAxesLE3;
            hold(ax1,"on");
            hold(ax3,"on");
            
            ft = fittype('k*x+d');
            excludedData = yLocs > thickness*2/3;
            opts = fitoptions('Method','NonlinearLeastSquares');
            opts.Exclude = excludedData;

            [f, fitness] = fit(xLocs',yLocs',ft,opts);
            k = f.k;
            d = f.d;
            frs = fitness.rsquare;
            frm = fitness.rmse;

            app.kmmsEditField.Value = k;
            app.dmmEditField.Value = d;
            app.R2EditField.Value = frs;
            app.RMSEEditField.Value = frm;
                
            fitData = feval(f,xData);
            plot(ax1,xData,fitData,'m--');
            plot(ax3,xData,fitData,'m--');
            legend(ax3,"Liquid-front","Fitting Function","Location","southeast")

            if app.Upto3mmCheckBox.Value
                ylim(ax1,[0,3]);
                ylim(ax3,[0,3]);
            end
            
            % Calculate the liquid ingress time in seconds
            liquidIngressTime = (thickness/2-d)/k;
            liquidIngressTime = floor(liquidIngressTime*100)/100;
            app.LiquidIngressTimesecEditField.Value = liquidIngressTime;            
        end

        % Button pushed function: SaveTruncatedButton
        function SaveTruncatedButtonPushed(app, event)
            app.stopProcess = 0;

            try
                samData = app.TData.samData;
                xData = app.TData.xData;
                ToF = app.TData.ToF;
            catch ME
                return
            end

            filter = {'*.thz';'*.*'};
            [filename, filepath] = uiputfile(filter);
            
            if isequal(filename,0)||isequal(filepath,0)
                return;          
            end

            newfullfile = strcat(filepath,filename);
            delete(newfullfile);
            measNum = size(samData,1);

            app.SystemStatusEditField.Value = 'Exporting truncated dataset....';
            drawnow
                                              
            measNum = size(samData,2);
            
            %Y-trancation option
            lbLoc = 1;
            ubLoc = measNum;
            
            if app.LowerEditField.Value
                lb = app.LowerEditField.Value;
                lbLoc = sum(xData <= lb);              
            end
            
            if app.UpperEditField.Value
                ub = app.UpperEditField.Value;
                ubLoc = sum(xData <=ub);
            end
            
            % lb, ub availability check
            if lbLoc==0 || ubLoc>measNum || lbLoc>ubLoc
                fig = app.DipTabUIFigure;
                uialert(fig,'Incorrect Y-truncation setting','Warning');
                 app.DeployButton.Enable = true;
                app.TruncateButton.Enable = true;
                app.SystemStatusEditField.Value = 'Exporting cancelled';
                return;
            end

            oldfullfile = app.filefullpath;
            
            if isempty(oldfullfile)
                     return;
            end
            
           
            % import .thz file to the workspace
            thzInfo = h5info(oldfullfile);
            oldeasNum = size(thzInfo.Groups,1);
            ListItems = cell(oldeasNum,1);
            [ListItems{:}] = deal(thzInfo.Groups.Name);
            %assignin("base","thzInfo",thzInfo);


            attrItems = ["description","instrument","user","time","mode","coordinates","mdDescription",...
                "md1","md2","md3","md4","thzVer","dsDescription"];
            dn1 = ListItems{1};
            cnt = 1;

            for idx = lbLoc:ubLoc
                dn = ListItems{idx};
                dsn = strcat(dn,'/ds1');
                ds = h5read(oldfullfile,dsn);
                h5create(newfullfile,dsn,size(ds));
                h5write(newfullfile,dsn,ds);

                timeAtt = h5readatt(oldfullfile,dn,"time");
                h5writeatt(newfullfile,dn,"time",timeAtt);

                if isequal(idx,lbLoc)
                    for attrName = attrItems
                        try
                            attData = h5readatt(oldfullfile,dn1,attrName);
                            h5writeatt(newfullfile,dn,attrName,attData);
                        catch ME
                        end
                    end
                end

                progressP = cnt/(ubLoc-lbLoc)*100;
                progressP = num2str(progressP,'%.0f');
                progressP = strcat("Exporting: ", progressP,"%");
                app.SystemStatusEditField.Value = progressP;
                drawnow
                cnt = cnt + 1;

                if app.stopProcess
                    app.SystemStatusEditField.Value = "Process aborted";
                    app.SpectrogramButton.Enable = true;
                    return
                end
            end

            app.SystemStatusEditField.Value = "Exporting finished";

           
        end

        % Button pushed function: AssiginFFTDatainworkspaceButton
        function AssiginFFTDatainworkspaceButtonPushed(app, event)
            try
                FData = app.FData
            catch ME
                return
            end

            assignin("base","FData",FData);
        end

        % Button pushed function: DrawPolylineButton
        function DrawPolylineButtonPushed(app, event)
            fig = app.DipTabUIFigure;

            try
                samData = app.TData.samData;
                xData = app.TData.xData;
                displacement = app.TData.displacement;
                thickness = app.ThicknessmmEditField_LE.Value;
            catch ME
                uialert(fig,'Dataset is not ready','Warning');
                app.EnableButton.Value = false;
                return
            end
            
            cmap = app.ExtColormapDropDown.Value;
            alp = str2num(app.AlphaDropDown.Value);
            ROIwidth = app.ROIwidthEditField.Value;
            
            ax1 = app.UIAxesLE1;
            ax2 = app.UIAxesLE2;
            ax3 = app.UIAxesLE3;

            hold(ax1,"on");
            hold(ax3,"on");
            
            app.SystemStatusEditField.Value = "Please draw a polyline using mouse on the terahertz reflectometry plot";

            % Try to draw a polyline and catch any errors
            try 
                roi = drawpolyline(ax1,"Color","r");

                % Check if the ROI object is valid and contains the
                % necessary data
                if isempty(roi) || isempty(roi.Position)
                    uialert(fig,'Invalid ROI Selection','Warning');
                    disp('Invalid ROI, Selection Cancelled');
                    return;
                else
                    disp('ROI selected sucessfully.');
                end
            catch ME
                % Handle the error
                uialert(fig,'ROI Selection Error','Warning');
                return;
            end

            ROImat = createMask(roi);
            K1 = gausswin(ROIwidth);
            ROImat = conv2(ROImat,K1,"same");
            ROImat = ROImat ~= 0;
            ROIvec = sum(ROImat,1);
            ROIvec = ROIvec ~= 0;
            colInitNum = find(ROIvec,1); % first non-zero column number
            colNum = colInitNum;

            K2 = gausswin(ceil(ROIwidth/2));
            samData = conv2(samData,K2,"same");
            samDataROI = samData.*ROImat;
            eFiledAmp = [];
            yLocs = [];

            while(ROIvec(colNum))
                %[pks, locs] = findpeaks(samDataROI(:,colNum));
                [pks, locs] = max(samDataROI(:,colNum));
                eFiledAmp = [eFiledAmp pks];
                yLocs = [yLocs displacement(locs)];
                colNum = colNum + 1;
            end

            colVec = (colInitNum:colNum-1);
            xLocs = xData(colVec);

            % trim points that have its y-value over 2/3 of the thickness
            xLocs(yLocs > thickness*2/3) = [];
            eFiledAmp(yLocs > thickness*2/3) = [];
            yLocs(yLocs > thickness*2/3) = [];

            plot(ax1,xLocs,yLocs,'.');            
            plot(ax2,xLocs,eFiledAmp);
            plot(ax3,xLocs,yLocs,'.');

            colormap(ax1,cmap);
            xlim(ax2,[xData(1) xData(end)]);
            xlim(ax3,[xData(1) xData(end)]);
            ylim(ax3,[displacement(end) displacement(1)]);
            axis(ax1,'tight');

            if app.DisplayTabletCentreCheckBox.Value
                yline(ax3,thickness/2,'--','Centre Line');
            end

            if app.Upto3mmCheckBox.Value
                ylim(ax1,[0,3]);
                ylim(ax3,[0,3]);
            end

            Peaks.xLocs = xLocs;
            Peaks.yLocs = yLocs;
            Peaks.eFieldAmp = eFiledAmp;
            app.TData.Peaks = Peaks;

            app.TData.fhROI = ROImat;
            app.SystemStatusEditField.Value = "Liquidfront points are selected.";
        end

        % Button pushed function: PlotButton_DC3D
        function PlotButton_DC3DPushed(app, event)
            % Create UIFigure and hide until all components are created
            fig = figure('Visible', 'on');
            fig.Position = [100 100 1200 800];
            fig.Name = "DipTab 3D Plot";

            % Create UIAxes
            ax = uiaxes(fig);
            axis(ax,'tight');
            grid(ax,"on");
            hold(ax,'on');
            box(ax,"on");
            xlabel(ax, 'Time (sec)');
            ylabel(ax, 'Displacement (ps)');
            zlabel(ax, 'E field (a.u.)');
            ax.Position = [20 10 1140 780];
            itemNums = app.MeasurementListBox.Value;
            itemNames = app.MeasurementListBox.Items;
            
            if isempty(itemNums)
                fig = app.DipTabUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            Peaks = app.CData.Peaks;            
            lh1 = [];
            lh = [];
            cnt = 1;
            
            for idx = itemNums
                h = plot3(ax,Peaks{idx}.xLocs,Peaks{idx}.yLocs,Peaks{idx}.eFieldAmp,'.');
                lh(cnt) = h;
                
                cnt = cnt + 1;
            end
                        
            if app.LegendCheckBox.Value
                legend(ax,lh,itemNames([itemNums]),"Location","southeast","Interpreter","none");
            else
                legend(ax,"off");
            end
        end

        % Button pushed function: PlotButton_DCNew
        function PlotButton_DCNewPushed(app, event)
            % Create UIFigure and hide until all components are created
            fig = uifigure('Visible', 'on');
            fig.Position = [100 100 1200 900];
            fig.Name = "DipTab Colour Plot";

            % Create UIAxes
            ax = uiaxes(fig);
            axis(ax,'tight');
            grid(ax,"on");
            hold(ax,'on');
            box(ax,"on");
            colormap(ax,"jet");
            xlabel(ax, 'Time (sec)');
            ylabel(ax, 'Displacement (ps)');
            colorbar(ax,"northoutside");
            %zlabel(ax, 'E field (a.u.)');
            ax.Position = [20 10 1140 780];
            itemNums = app.MeasurementListBox.Value;
            itemNames = app.MeasurementListBox.Items;
            
            if isempty(itemNums)
                fig = app.DipTabUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            Peaks = app.CData.Peaks;            
            lh1 = [];
            lh = [];
            cnt = 1;
            
            for idx = itemNums
                h = scatter(ax,Peaks{idx}.xLocs,Peaks{idx}.yLocs,[],Peaks{idx}.eFieldAmp,'.');
                lh(cnt) = h;
                cnt = cnt + 1;
            end
                        
            if app.LegendCheckBox.Value
                legend(ax,lh,itemNames([itemNums]),"Location","southeast","Interpreter","none");
            else
                legend(ax,"off");
            end
        end

        % Button pushed function: DeployButton
        function DeployButtonPushed(app, event)
            fullfile = app.filefullpath;
            clearMemory(app);
            resetGeneralInfo(app);
            app.plotUpdate = true;
            ax = app.UIAxesTD1;
                        
            if isempty(fullfile)
                     return;
            end

            try
                question = "Select x-axis unit";
                xUnit = questdlg('Select x-axis unit','X-axis Unit','Time (sec)','Position (mm)','Time (sec)');
            catch
                return;
            end

            try
                question = "Select Peak Polarity";
                peakOption = questdlg('Is the referencing peak positive?','Peak Polarity','Positive','Negative','Positive');                    
            catch
                return;
            end

            if isequal(xUnit,'Time (sec)')
                xlabel(ax,'Time (sec)');
            else
                xlabel(ax,'Position (mm)');
            end
            
            app.SystemStatusEditField.Value = 'Loading....';
            drawnow
            
            % import .thz file to the workspace
            thzInfo = h5info(fullfile);
            measNum = size(thzInfo.Groups,1);
            ListItems = cell(measNum,1);
            %assignin("base","thzInfo",thzInfo)
            [ListItems{:}] = deal(thzInfo.Groups.Name);

            % estimate measurement time and set y-axis time interval
            srtGrp = ListItems{1};
            endGrp = ListItems{end};
            srtTime = h5readatt(fullfile,srtGrp,"time"); % extract measurement start time (string)
            endTime = h5readatt(fullfile,endGrp,"time"); % extract measurement end time (string)

            if contains(srtTime,'_')
                srtTime = extractAfter(srtTime,'_');
                endTime = extractAfter(endTime,'_');
            elseif contains(srtTime,' ')
                srtTime = extractAfter(srtTime,' ');
                endTime = extractAfter(endTime,' ');
            end

            if contains(srtTime,'.')
                srtTime = extractBefore(srtTime,'.');
                endTime = extractBefore(endTime,'.');
            end

            if contains(srtTime,'-')
                delimiter = '-';
            else
                delimiter = ':';
            end
                
            srtTime = [3600 60 1]*str2num(cell2mat(split(srtTime,delimiter))); % start time in seconds
            endTime = [3600 60 1]*str2num(cell2mat(split(endTime,delimiter))); % end time in seconds
            timeDiff = endTime - srtTime;

            if timeDiff < 0
                timeDiff = timeDiff + 86400;
            end

            % extract setting parameters
            srtDn = strcat(srtGrp,'/ds1');
            firstDataset = h5read(fullfile,srtDn);
            dataLength = size(firstDataset,2);
            xSpacing = mean(diff(firstDataset(1,:)));
            description = h5readatt(fullfile,srtGrp,"description");
            instrument = h5readatt(fullfile,srtGrp,"instrument");
            user = h5readatt(fullfile,srtGrp,"user");
            thickness = h5readatt(fullfile,srtGrp,"md1");

            try
                n_eff = h5readatt(fullfile,srtGrp,"md2");
            catch
                n_eff = 1.6;
            end

            if isequal(instrument,"<missing>")
                instrument = '';
            end

            % x, y axis units
            xDataItv = timeDiff/measNum;
            app.TData.xData = xDataItv*(0:measNum-1);
            ToF = xSpacing*(0:dataLength-1);
            app.TData.ToF = ToF;
            rawData = zeros(dataLength,measNum);

            % Scan information panel display
            app.DataLengthEditField.Value = dataLength;
            app.DataNumberEditField.Value = measNum;
            app.ToFSpacingpsEditField.Value = xSpacing;
            app.TimeSpacingsEditField.Value = xDataItv;
            app.ThicknessmmEditField.Value = thickness;
            app.RefractiveIndexEditField.Value = n_eff;
            % app.InstrumentEditField.Value = instrument;
            % app.UserEditField.Value = user;
            app.SampleDescriptionEditField.Value = description;
                                    
            % measurement dataset extraction
            for idx = 1:measNum
                dn = strcat(ListItems{idx},'/ds1');
                measData = h5read(fullfile,dn);

                if peakOption == "Negative"
                    rawData(:,idx) = measData(2,:)'*-1;
                else
                    rawData(:,idx) = measData(2,:)';
                end                
               
                progressP = idx/measNum*100;
                progressP = num2str(progressP,'%.0f');
                progressP = strcat("Loading: ", progressP,"%");
                app.SystemStatusEditField.Value = progressP;
                drawnow
            end

            
            baselineVector = rawData(:,1);
            rawData = rawData - baselineVector;

            app.TData.samData = rawData;
            app.TData.rawData = rawData;
            app.SystemStatusEditField.Value = 'Done';
            drawnow

            [cmin cmax] = bounds(rawData,"all");

            cmin = round(cmin*10^2)*10^-2;
            cmax = round(cmax*10^2)*10^-2;

            app.DataRangeFromEditField.Value = cmin;
            app.DataRangeToEditField.Value = cmax;
            app.AOIRangeFromEditField.Value = cmin;
            app.AOIRangeToEditField.Value = cmax;

            % x-axis unit conversion
            % if ~isempty(refractiveIndex)||refractiveIndex > 1
            %     xUnitCal(app);
            % end
            
            TDPlot(app); % painting 2D image display

            %app.DeployButton.Enable = true;
            app.EnableButton.Value = false;
            EnableButtonValueChanged(app);
            app.TruncateButton.Enable = true;
            app.TabGroup.SelectedTab = app.TabGroup.Children(1);
        end

        % Button pushed function: FrequencyDomainStudyButton
        function FrequencyDomainStudyButtonPushed(app, event)
            samData = app.TData.samData;
            dataLength=size(samData,1);
            measNum = size(samData,2);
            app.dataLengthEditField_FD.Value = dataLength;
            app.dataNumberEditField_FD.Value = measNum;
            app.DeployButton.Enable = true;
            app.TabGroup.SelectedTab = app.TabGroup.Children(4);
        end

        % Button pushed function: RemoveBaseButton_Tab1
        function RemoveBaseButton_Tab1Pushed(app, event)
            Data = app.TData;
            xData = Data.xData;
            ToF = Data.ToF;
            samData = Data.samData;
            
            samDataAvg = mean(samData,2);
            samData = samData - samDataAvg;
            app.TData.samData = samData;
            TDPlot(app);
        end

        % Value changed function: tYPickEditField
        function tYPickEditFieldValueChanged(app, event)
            value = app.tYPickEditField.Value;
            app.tYPickSlider.Value = value;
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Value changed function: tYPickSlider
        function tYPickSliderValueChanged(app, event)
            value = app.tYPickSlider.Value;
            app.tYPickEditField.Value = value;
            
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Value changing function: tYPickSlider
        function tYPickSliderValueChanging(app, event)
            changingValue = event.Value;
            app.tYPickEditField.Value = changingValue;
            
            if app.EnableButton.Value
                posT1_XYLine(app);
            end
        end

        % Button pushed function: PlotButton_3
        function PlotButton_3Pushed(app, event)
            plotSpectra(app);
            plotPhases(app);
        end

        % Value changed function: LocationSlider
        function LocationSliderValueChanged(app, event)
            posT2_XLine(app);
        end

        % Value changing function: LocationSlider
        function LocationSliderValueChanging(app, event)
            %changingValue = event.Value;
            posT2_XLine(app);
        end

        % Button pushed function: SetDownLimitButton
        function SetDownLimitButtonPushed(app, event)
            value = app.tYPickEditField.Value;
            app.DownEditField.Value = value;
        end

        % Button pushed function: SetLeftLimitButton
        function SetLeftLimitButtonPushed(app, event)
            value = app.tXPickEditField.Value;
            app.LeftEditField.Value = value;
        end

        % Button pushed function: ApplyButton
        function ApplyButtonPushed(app, event)
            samData = app.TData.samData;            
            N = app.gausswinEditField.Value;
            K = gausswin(N);
            Zsmooth = conv2(samData,K,'same');
            app.TData.samData = Zsmooth;
            TDPlot(app);
        end

        % Button pushed function: LoadRawDataButton
        function LoadRawDataButtonPushed(app, event)
            rawData = app.TData.rawData;
            app.TData.samData = rawData;
            TDPlot(app);
        end

        % Button pushed function: LiquidfrontExtractionButton
        function LiquidfrontExtractionButtonPushed(app, event)
            try
                samData = app.TData.samData;
                ToF = app.TData.ToF;
                app.TData.algoROI = [];
            catch ME
                fig = app.DipTabUIFigure;
                uialert(fig,'Datasets are not ready','Warning');
                return;
            end

            n_eff = app.RefractiveIndexEditField.Value;
            inAng = app.inAng;
            c = 3*10^8;

            displacement = ToF*10^-12.*c*sqrt(n_eff^2-sin(inAng)^2)/(2*n_eff^2);
            displacement = displacement * 10^3; % meter to millimeter
            displacement = flip(displacement);
            app.TData.displacement = displacement;

            app.ThicknessmmEditField_LE.Value = app.ThicknessmmEditField.Value;
            app.RefractiveIndexEditField_LE.Value = app.RefractiveIndexEditField.Value;
            
            app.dataLengthEditField_LE.Value = size(samData,1);
            app.dataNumberEditField_LE.Value = size(samData,2);
            app.SampleNameEditField.Value = app.ProjectNameEditField.Value;

            LFPlot(app);          
            app.TabGroup.SelectedTab = app.TabGroup.Children(2);
        end

        % Button pushed function: ExportTableasCSVFormatButton
        function ExportTableasCSVFormatButtonPushed(app, event)
            % Open a file save dialog box with a filter for CSV files
            filter = {'*.csv';'*.*'};
            [filename, filepath] = uiputfile(filter);
            
            % Check if the user selected a file or cancelled the dialog
            if isequal(filename, 0) || isequal(filepath, 0)
                return;
            end
            
            % Combine the file path and file name
            fullFileName = fullfile(filepath, filename);
            
            % Retrieve the cell array from the app structure
            Tcell = app.Tcell;
            
            % Convert the cell array to a table and specify the column names
            T = cell2table(Tcell, 'VariableNames', {'Sample Name', 'Thickness (mm)', 'Refractive Index', 'Ingress Time (s)','k','d', 'R^2','RMSE' });

            % Write the table to a CSV file
            writetable(T, fullFileName);
        end

        % Button pushed function: SaveFigureButton_LE
        function SaveFigureButton_LEPushed(app, event)
            ax = app.UIAxesLE1;
            sampleName = app.SampleNameEditField.Value;
            sampleName = strcat(sampleName,'.pdf');
            filter = {'*.pdf';'*.*'};
            [filename, filepath] = uiputfile(filter,'Save as',sampleName);
            
            if isequal(filename,0)||isequal(filepath,0)
                return;          
            end
            
            fullfile = strcat(filepath,filename);
            exportgraphics(ax,fullfile);
        end

        % Button pushed function: UpButton
        function UpButtonPushed(app, event)
            itemNum = app.MeasurementListBox.Value;
            ListBoxItems = app.MeasurementListBox.Items;
            fig = app.DipTabUIFigure;

            if length(itemNum) ~= 1 || itemNum == 1
                uialert(fig,'Invalid Selection(s)','Process Aborted');
                return;
            end          
                        
            PeaksTemp = app.CData.Peaks(itemNum-1);
            metaTemp = app.CData.meta(itemNum-1);
            batchTemp = app.CData.batch(itemNum-1);
            ListBoxItemsTemp = ListBoxItems(itemNum-1);
            TcellTemp = app.Tcell(itemNum-1,:);

            app.CData.Peaks(itemNum-1) = app.CData.Peaks(itemNum);
            app.CData.meta(itemNum-1) = app.CData.meta(itemNum);
            app.CData.batch(itemNum-1) = app.CData.batch(itemNum);
            ListBoxItems(itemNum-1) = ListBoxItems(itemNum);
            app.Tcell(itemNum-1,:) = app.Tcell(itemNum,:);

            app.CData.Peaks(itemNum) = PeaksTemp;
            app.CData.meta(itemNum) = metaTemp;
            app.CData.batch(itemNum) = batchTemp;
            ListBoxItems(itemNum) = ListBoxItemsTemp;
            app.Tcell(itemNum,:) = TcellTemp;

            app.MeasurementListBox.Value = itemNum-1;

            app.MeasurementListBox.Items = ListBoxItems;
            app.MeasurementListBox.ItemsData = (1:length(ListBoxItems));
            updateTable(app);
        end

        % Button pushed function: DownButton
        function DownButtonPushed(app, event)
            itemNum = app.MeasurementListBox.Value;
            ListBoxItems = app.MeasurementListBox.Items;
            fig = app.DipTabUIFigure;

            if length(itemNum) ~= 1 || itemNum == length(ListBoxItems)
                uialert(fig,'Invalid Selection(s)','Process Aborted');
                return;
            end          

            PeaksTemp = app.CData.Peaks(itemNum+1);
            metaTemp = app.CData.meta(itemNum+1);
            batchTemp = app.CData.batch(itemNum+1);
            ListBoxItemsTemp = ListBoxItems(itemNum+1);
            TcellTemp = app.Tcell(itemNum+1,:);

            app.CData.Peaks(itemNum+1) = app.CData.Peaks(itemNum);
            app.CData.meta(itemNum+1) = app.CData.meta(itemNum);
            app.CData.batch(itemNum+1) = app.CData.batch(itemNum);
            ListBoxItems(itemNum+1) = ListBoxItems(itemNum);
            app.Tcell(itemNum+1,:) = app.Tcell(itemNum,:);

            app.CData.Peaks(itemNum) = PeaksTemp;
            app.CData.meta(itemNum) = metaTemp;
            app.CData.batch(itemNum) = batchTemp;
            ListBoxItems(itemNum) = ListBoxItemsTemp;
            app.Tcell(itemNum,:) = TcellTemp;

            app.MeasurementListBox.Value = itemNum+1;

            app.MeasurementListBox.Items = ListBoxItems;
            app.MeasurementListBox.ItemsData = (1:length(ListBoxItems));
            updateTable(app);
        end

        % Cell edit callback: UITable
        function UITableCellEdit(app, event)
            indices = event.Indices;
            newData = event.NewData;
            ListBoxItems = app.MeasurementListBox.Items;

            itemNum = indices(1);
            app.CData.meta(itemNum).sampleName = newData;
            ListBoxItems(itemNum) = {newData};
            app.Tcell(itemNum,1) = {newData};
            app.MeasurementListBox.Items = ListBoxItems;            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create DipTabUIFigure and hide until all components are created
            app.DipTabUIFigure = uifigure('Visible', 'off');
            app.DipTabUIFigure.Position = [100 50 1484 916];
            app.DipTabUIFigure.Name = 'DipTab';
            app.DipTabUIFigure.Icon = fullfile(pathToMLAPP, 'Images', 'icon.png');
            app.DipTabUIFigure.Scrollable = 'on';

            % Create ImportthzFileButton
            app.ImportthzFileButton = uibutton(app.DipTabUIFigure, 'push');
            app.ImportthzFileButton.ButtonPushedFcn = createCallbackFcn(app, @ImportthzFileButtonPushed, true);
            app.ImportthzFileButton.FontSize = 14;
            app.ImportthzFileButton.FontWeight = 'bold';
            app.ImportthzFileButton.Position = [300 851 128 28];
            app.ImportthzFileButton.Text = 'Import .thz File';

            % Create TerahertzLqiuidFrontDateAnalyserLabel
            app.TerahertzLqiuidFrontDateAnalyserLabel = uilabel(app.DipTabUIFigure);
            app.TerahertzLqiuidFrontDateAnalyserLabel.FontWeight = 'bold';
            app.TerahertzLqiuidFrontDateAnalyserLabel.FontAngle = 'italic';
            app.TerahertzLqiuidFrontDateAnalyserLabel.Position = [84 840 224 29];
            app.TerahertzLqiuidFrontDateAnalyserLabel.Text = 'Terahertz LqiuidFront Date Analyser';

            % Create ProjectNameEditField
            app.ProjectNameEditField = uieditfield(app.DipTabUIFigure, 'text');
            app.ProjectNameEditField.Editable = 'off';
            app.ProjectNameEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.ProjectNameEditField.Position = [440 854 432 22];

            % Create SystemStatusEditFieldLabel
            app.SystemStatusEditFieldLabel = uilabel(app.DipTabUIFigure);
            app.SystemStatusEditFieldLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.SystemStatusEditFieldLabel.HorizontalAlignment = 'right';
            app.SystemStatusEditFieldLabel.Position = [30 12 83 22];
            app.SystemStatusEditFieldLabel.Text = 'System Status';

            % Create SystemStatusEditField
            app.SystemStatusEditField = uieditfield(app.DipTabUIFigure, 'text');
            app.SystemStatusEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.SystemStatusEditField.Position = [128 12 822 22];

            % Create TabGroup
            app.TabGroup = uitabgroup(app.DipTabUIFigure);
            app.TabGroup.Position = [19 46 1446 790];

            % Create TimeDomainTab
            app.TimeDomainTab = uitab(app.TabGroup);
            app.TimeDomainTab.Title = 'Time Domain';

            % Create UIAxesTD1
            app.UIAxesTD1 = uiaxes(app.TimeDomainTab);
            title(app.UIAxesTD1, '2D Image of Terahertz Reflectometry')
            xlabel(app.UIAxesTD1, 'Time/Position (sec/mm)')
            ylabel(app.UIAxesTD1, 'Time of Flight (ps)')
            app.UIAxesTD1.PlotBoxAspectRatio = [1 1.13357400722022 1];
            app.UIAxesTD1.FontWeight = 'bold';
            app.UIAxesTD1.XTickLabelRotation = 0;
            app.UIAxesTD1.YTickLabelRotation = 0;
            app.UIAxesTD1.ZTickLabelRotation = 0;
            app.UIAxesTD1.Box = 'on';
            app.UIAxesTD1.FontSize = 12;
            app.UIAxesTD1.Position = [237 42 640 710];

            % Create UIAxesTD2
            app.UIAxesTD2 = uiaxes(app.TimeDomainTab);
            title(app.UIAxesTD2, 'Single E-field')
            xlabel(app.UIAxesTD2, 'Time of Flight (ps)')
            ylabel(app.UIAxesTD2, 'E field (a.u.)')
            app.UIAxesTD2.FontWeight = 'bold';
            app.UIAxesTD2.Box = 'on';
            app.UIAxesTD2.FontSize = 12;
            app.UIAxesTD2.Position = [881 539 560 220];

            % Create UIAxesTD3
            app.UIAxesTD3 = uiaxes(app.TimeDomainTab);
            title(app.UIAxesTD3, 'Cummurative E-field')
            xlabel(app.UIAxesTD3, 'Time/Position (sec/mm)')
            ylabel(app.UIAxesTD3, 'E field (a.u.)')
            app.UIAxesTD3.FontWeight = 'bold';
            app.UIAxesTD3.Box = 'on';
            app.UIAxesTD3.FontSize = 12;
            app.UIAxesTD3.Position = [883 262 560 220];

            % Create UIAxesTD4
            app.UIAxesTD4 = uiaxes(app.TimeDomainTab);
            title(app.UIAxesTD4, 'Spectrogram / Spectrum')
            xlabel(app.UIAxesTD4, 'Time (ps) / Frequency (THz)')
            ylabel(app.UIAxesTD4, 'Frequency (THz) / E field')
            zlabel(app.UIAxesTD4, 'Z')
            app.UIAxesTD4.FontWeight = 'bold';
            app.UIAxesTD4.Box = 'on';
            app.UIAxesTD4.FontSize = 12;
            app.UIAxesTD4.Position = [883 6 560 200];

            % Create GeneralInformationPanel
            app.GeneralInformationPanel = uipanel(app.TimeDomainTab);
            app.GeneralInformationPanel.Title = 'General Information';
            app.GeneralInformationPanel.Position = [16 583 216 160];

            % Create DataLengthLabel
            app.DataLengthLabel = uilabel(app.GeneralInformationPanel);
            app.DataLengthLabel.HorizontalAlignment = 'right';
            app.DataLengthLabel.Position = [3 111 70 22];
            app.DataLengthLabel.Text = 'Data Length';

            % Create DataLengthEditField
            app.DataLengthEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.DataLengthEditField.ValueDisplayFormat = '%.0f';
            app.DataLengthEditField.Position = [77 111 41 22];

            % Create DataNumberLabel
            app.DataNumberLabel = uilabel(app.GeneralInformationPanel);
            app.DataNumberLabel.HorizontalAlignment = 'right';
            app.DataNumberLabel.Position = [119 111 48 22];
            app.DataNumberLabel.Text = 'Number';

            % Create DataNumberEditField
            app.DataNumberEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.DataNumberEditField.ValueDisplayFormat = '%.0f';
            app.DataNumberEditField.Position = [169 111 41 22];

            % Create XSpacingpsEditFieldLabel
            app.XSpacingpsEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.XSpacingpsEditFieldLabel.Position = [7 84 99 22];
            app.XSpacingpsEditFieldLabel.Text = 'ToF Spacing (ps) ';

            % Create ToFSpacingpsEditField
            app.ToFSpacingpsEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.ToFSpacingpsEditField.ValueDisplayFormat = '%5.3f';
            app.ToFSpacingpsEditField.Position = [140 84 70 22];

            % Create TimeSpacingsEditFieldLabel
            app.TimeSpacingsEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.TimeSpacingsEditFieldLabel.HorizontalAlignment = 'right';
            app.TimeSpacingsEditFieldLabel.Position = [2 58 95 22];
            app.TimeSpacingsEditFieldLabel.Text = 'Time Spacing (s)';

            % Create TimeSpacingsEditField
            app.TimeSpacingsEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.TimeSpacingsEditField.ValueDisplayFormat = '%5.2f';
            app.TimeSpacingsEditField.Position = [140 58 70 22];

            % Create ThicknessmmEditFieldLabel
            app.ThicknessmmEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.ThicknessmmEditFieldLabel.HorizontalAlignment = 'right';
            app.ThicknessmmEditFieldLabel.Position = [3 32 89 22];
            app.ThicknessmmEditFieldLabel.Text = 'Thickness (mm)';

            % Create ThicknessmmEditField
            app.ThicknessmmEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.ThicknessmmEditField.Limits = [0 Inf];
            app.ThicknessmmEditField.ValueDisplayFormat = '%5.2f';
            app.ThicknessmmEditField.Position = [140 31 70 22];

            % Create RefractiveIndexEditFieldLabel
            app.RefractiveIndexEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.RefractiveIndexEditFieldLabel.HorizontalAlignment = 'right';
            app.RefractiveIndexEditFieldLabel.Position = [3 6 92 22];
            app.RefractiveIndexEditFieldLabel.Text = 'Refractive Index';

            % Create RefractiveIndexEditField
            app.RefractiveIndexEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.RefractiveIndexEditField.Limits = [0 Inf];
            app.RefractiveIndexEditField.ValueDisplayFormat = '%5.2f';
            app.RefractiveIndexEditField.Position = [140 5 70 22];
            app.RefractiveIndexEditField.Value = 1.5;

            % Create GuidelinesPanel
            app.GuidelinesPanel = uipanel(app.TimeDomainTab);
            app.GuidelinesPanel.Title = 'Guidelines';
            app.GuidelinesPanel.Position = [16 303 216 91];

            % Create EnableButton
            app.EnableButton = uibutton(app.GuidelinesPanel, 'state');
            app.EnableButton.ValueChangedFcn = createCallbackFcn(app, @EnableButtonValueChanged, true);
            app.EnableButton.Text = 'Enable';
            app.EnableButton.Position = [10 38 92 23];

            % Create SpectrogramButton
            app.SpectrogramButton = uibutton(app.GuidelinesPanel, 'push');
            app.SpectrogramButton.ButtonPushedFcn = createCallbackFcn(app, @SpectrogramButtonPushed, true);
            app.SpectrogramButton.Enable = 'off';
            app.SpectrogramButton.Position = [10 9 92 23];
            app.SpectrogramButton.Text = 'Spectrogram';

            % Create PowerSpectrumButton
            app.PowerSpectrumButton = uibutton(app.GuidelinesPanel, 'push');
            app.PowerSpectrumButton.ButtonPushedFcn = createCallbackFcn(app, @PowerSpectrumButtonPushed, true);
            app.PowerSpectrumButton.Enable = 'off';
            app.PowerSpectrumButton.Position = [107 10 100 23];
            app.PowerSpectrumButton.Text = 'Power Spectrum';

            % Create AutoScanSpectrumCheckBox
            app.AutoScanSpectrumCheckBox = uicheckbox(app.GuidelinesPanel);
            app.AutoScanSpectrumCheckBox.Text = 'Auto-Scan';
            app.AutoScanSpectrumCheckBox.Position = [123 39 79 22];

            % Create ColormapPanel
            app.ColormapPanel = uipanel(app.TimeDomainTab);
            app.ColormapPanel.Title = 'Colormap';
            app.ColormapPanel.Position = [16 402 216 85];

            % Create ColormapDropDown
            app.ColormapDropDown = uidropdown(app.ColormapPanel);
            app.ColormapDropDown.Items = {'parula', 'jet', 'copper', 'bone', 'hot'};
            app.ColormapDropDown.ValueChangedFcn = createCallbackFcn(app, @ColormapDropDownValueChanged, true);
            app.ColormapDropDown.Position = [118 37 86 22];
            app.ColormapDropDown.Value = 'parula';

            % Create PlotButton
            app.PlotButton = uibutton(app.ColormapPanel, 'push');
            app.PlotButton.ButtonPushedFcn = createCallbackFcn(app, @PlotButtonPushed, true);
            app.PlotButton.Position = [9 9 198 22];
            app.PlotButton.Text = 'Plot';

            % Create colorbarCheckBox
            app.colorbarCheckBox = uicheckbox(app.ColormapPanel);
            app.colorbarCheckBox.Text = 'colorbar';
            app.colorbarCheckBox.Position = [50 37 66 22];

            % Create DCheckBox
            app.DCheckBox = uicheckbox(app.ColormapPanel);
            app.DCheckBox.Text = '3D';
            app.DCheckBox.Position = [11 37 37 22];

            % Create AOIBoundaryTruncationPanel
            app.AOIBoundaryTruncationPanel = uipanel(app.TimeDomainTab);
            app.AOIBoundaryTruncationPanel.Title = 'AOI Boundary Truncation';
            app.AOIBoundaryTruncationPanel.Position = [16 169 216 126];

            % Create TruncateButton
            app.TruncateButton = uibutton(app.AOIBoundaryTruncationPanel, 'push');
            app.TruncateButton.ButtonPushedFcn = createCallbackFcn(app, @TruncateButtonPushed, true);
            app.TruncateButton.BackgroundColor = [1 1 1];
            app.TruncateButton.FontWeight = 'bold';
            app.TruncateButton.FontColor = [0.149 0.149 0.149];
            app.TruncateButton.Position = [116 12 93 23];
            app.TruncateButton.Text = 'Truncate';

            % Create RemoveBaseButton_Tab1
            app.RemoveBaseButton_Tab1 = uibutton(app.AOIBoundaryTruncationPanel, 'push');
            app.RemoveBaseButton_Tab1.ButtonPushedFcn = createCallbackFcn(app, @RemoveBaseButton_Tab1Pushed, true);
            app.RemoveBaseButton_Tab1.Position = [116 71 93 23];
            app.RemoveBaseButton_Tab1.Text = 'Remove Base';

            % Create LeftEditFieldLabel
            app.LeftEditFieldLabel = uilabel(app.AOIBoundaryTruncationPanel);
            app.LeftEditFieldLabel.HorizontalAlignment = 'right';
            app.LeftEditFieldLabel.Position = [3 42 25 22];
            app.LeftEditFieldLabel.Text = 'Left';

            % Create LeftEditField
            app.LeftEditField = uieditfield(app.AOIBoundaryTruncationPanel, 'numeric');
            app.LeftEditField.Limits = [0 Inf];
            app.LeftEditField.ValueDisplayFormat = '%5.2f';
            app.LeftEditField.Position = [34 42 40 22];

            % Create RightEditFieldLabel
            app.RightEditFieldLabel = uilabel(app.AOIBoundaryTruncationPanel);
            app.RightEditFieldLabel.HorizontalAlignment = 'right';
            app.RightEditFieldLabel.Position = [126 41 33 22];
            app.RightEditFieldLabel.Text = 'Right';

            % Create RightEditField
            app.RightEditField = uieditfield(app.AOIBoundaryTruncationPanel, 'numeric');
            app.RightEditField.Limits = [0 Inf];
            app.RightEditField.ValueDisplayFormat = '%5.2f';
            app.RightEditField.Position = [81 42 43 22];

            % Create UpEditFieldLabel
            app.UpEditFieldLabel = uilabel(app.AOIBoundaryTruncationPanel);
            app.UpEditFieldLabel.HorizontalAlignment = 'right';
            app.UpEditFieldLabel.Position = [32 71 25 22];
            app.UpEditFieldLabel.Text = 'Up';

            % Create UpEditField
            app.UpEditField = uieditfield(app.AOIBoundaryTruncationPanel, 'numeric');
            app.UpEditField.ValueDisplayFormat = '%5.2f';
            app.UpEditField.Position = [61 71 40 22];

            % Create DownEditFieldLabel
            app.DownEditFieldLabel = uilabel(app.AOIBoundaryTruncationPanel);
            app.DownEditFieldLabel.HorizontalAlignment = 'right';
            app.DownEditFieldLabel.Position = [16 12 36 22];
            app.DownEditFieldLabel.Text = 'Down';

            % Create DownEditField
            app.DownEditField = uieditfield(app.AOIBoundaryTruncationPanel, 'numeric');
            app.DownEditField.ValueDisplayFormat = '%5.2f';
            app.DownEditField.Position = [58 12 43 22];

            % Create FrequencyDomainStudyButton
            app.FrequencyDomainStudyButton = uibutton(app.TimeDomainTab, 'push');
            app.FrequencyDomainStudyButton.ButtonPushedFcn = createCallbackFcn(app, @FrequencyDomainStudyButtonPushed, true);
            app.FrequencyDomainStudyButton.Position = [24 38 193 25];
            app.FrequencyDomainStudyButton.Text = 'Frequency Domain Study';

            % Create SaveTruncatedButton
            app.SaveTruncatedButton = uibutton(app.TimeDomainTab, 'push');
            app.SaveTruncatedButton.ButtonPushedFcn = createCallbackFcn(app, @SaveTruncatedButtonPushed, true);
            app.SaveTruncatedButton.Position = [24 8 193 25];
            app.SaveTruncatedButton.Text = 'AOI THz Save';

            % Create ColormapcontrolPanel_TD
            app.ColormapcontrolPanel_TD = uipanel(app.TimeDomainTab);
            app.ColormapcontrolPanel_TD.Title = 'Colormap control';
            app.ColormapcontrolPanel_TD.Position = [16 495 216 80];

            % Create DataRangeEditFieldLabel
            app.DataRangeEditFieldLabel = uilabel(app.ColormapcontrolPanel_TD);
            app.DataRangeEditFieldLabel.HorizontalAlignment = 'right';
            app.DataRangeEditFieldLabel.Position = [5 33 69 22];
            app.DataRangeEditFieldLabel.Text = 'Data Range';

            % Create DataRangeFromEditField
            app.DataRangeFromEditField = uieditfield(app.ColormapcontrolPanel_TD, 'numeric');
            app.DataRangeFromEditField.ValueDisplayFormat = '%5.2f';
            app.DataRangeFromEditField.Editable = 'off';
            app.DataRangeFromEditField.Position = [83 33 43 22];

            % Create Label
            app.Label = uilabel(app.ColormapcontrolPanel_TD);
            app.Label.HorizontalAlignment = 'right';
            app.Label.Position = [123 33 25 22];
            app.Label.Text = '-';

            % Create DataRangeToEditField
            app.DataRangeToEditField = uieditfield(app.ColormapcontrolPanel_TD, 'numeric');
            app.DataRangeToEditField.ValueDisplayFormat = '%5.2f';
            app.DataRangeToEditField.Editable = 'off';
            app.DataRangeToEditField.Position = [163 33 43 22];

            % Create DOIRangeLabel
            app.DOIRangeLabel = uilabel(app.ColormapcontrolPanel_TD);
            app.DOIRangeLabel.HorizontalAlignment = 'right';
            app.DOIRangeLabel.Position = [6 5 64 22];
            app.DOIRangeLabel.Text = 'AOI Range';

            % Create AOIRangeFromEditField
            app.AOIRangeFromEditField = uieditfield(app.ColormapcontrolPanel_TD, 'numeric');
            app.AOIRangeFromEditField.ValueDisplayFormat = '%5.2f';
            app.AOIRangeFromEditField.Position = [83 5 43 22];

            % Create Label_2
            app.Label_2 = uilabel(app.ColormapcontrolPanel_TD);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.Position = [124 5 25 22];
            app.Label_2.Text = '-';

            % Create AOIRangeToEditField
            app.AOIRangeToEditField = uieditfield(app.ColormapcontrolPanel_TD, 'numeric');
            app.AOIRangeToEditField.ValueDisplayFormat = '%5.2f';
            app.AOIRangeToEditField.Position = [164 5 43 22];

            % Create SaveFigureButton
            app.SaveFigureButton = uibutton(app.TimeDomainTab, 'push');
            app.SaveFigureButton.ButtonPushedFcn = createCallbackFcn(app, @SaveFigureButtonPushed, true);
            app.SaveFigureButton.Position = [722 15 121 23];
            app.SaveFigureButton.Text = 'Save Figure';

            % Create tXPickSlider
            app.tXPickSlider = uislider(app.TimeDomainTab);
            app.tXPickSlider.ValueChangedFcn = createCallbackFcn(app, @tXPickSliderValueChanged, true);
            app.tXPickSlider.ValueChangingFcn = createCallbackFcn(app, @tXPickSliderValueChanging, true);
            app.tXPickSlider.Enable = 'off';
            app.tXPickSlider.Position = [921 250 383 3];

            % Create YlinesecLabel
            app.YlinesecLabel = uilabel(app.TimeDomainTab);
            app.YlinesecLabel.HorizontalAlignment = 'right';
            app.YlinesecLabel.Enable = 'off';
            app.YlinesecLabel.Position = [1323 242 62 22];
            app.YlinesecLabel.Text = 'X line(sec)';

            % Create tXPickEditField
            app.tXPickEditField = uieditfield(app.TimeDomainTab, 'numeric');
            app.tXPickEditField.Limits = [0 Inf];
            app.tXPickEditField.ValueDisplayFormat = '%5.2f';
            app.tXPickEditField.ValueChangedFcn = createCallbackFcn(app, @tXPickEditFieldValueChanged, true);
            app.tXPickEditField.Enable = 'off';
            app.tXPickEditField.Position = [1386 242 47 22];

            % Create tYPickSlider
            app.tYPickSlider = uislider(app.TimeDomainTab);
            app.tYPickSlider.ValueChangedFcn = createCallbackFcn(app, @tYPickSliderValueChanged, true);
            app.tYPickSlider.ValueChangingFcn = createCallbackFcn(app, @tYPickSliderValueChanging, true);
            app.tYPickSlider.Enable = 'off';
            app.tYPickSlider.Position = [921 523 383 3];

            % Create tYPickEditField
            app.tYPickEditField = uieditfield(app.TimeDomainTab, 'numeric');
            app.tYPickEditField.Limits = [0 Inf];
            app.tYPickEditField.ValueDisplayFormat = '%5.2f';
            app.tYPickEditField.ValueChangedFcn = createCallbackFcn(app, @tYPickEditFieldValueChanged, true);
            app.tYPickEditField.Enable = 'off';
            app.tYPickEditField.Position = [1385 519 47 22];

            % Create XlinesecLabel
            app.XlinesecLabel = uilabel(app.TimeDomainTab);
            app.XlinesecLabel.HorizontalAlignment = 'right';
            app.XlinesecLabel.Enable = 'off';
            app.XlinesecLabel.Position = [1321 519 61 22];
            app.XlinesecLabel.Text = 'Y line(sec)';

            % Create SetLeftLimitButton
            app.SetLeftLimitButton = uibutton(app.TimeDomainTab, 'push');
            app.SetLeftLimitButton.ButtonPushedFcn = createCallbackFcn(app, @SetLeftLimitButtonPushed, true);
            app.SetLeftLimitButton.Enable = 'off';
            app.SetLeftLimitButton.Position = [1337 214 100 23];
            app.SetLeftLimitButton.Text = 'Set Left Limit';

            % Create SetDownLimitButton
            app.SetDownLimitButton = uibutton(app.TimeDomainTab, 'push');
            app.SetDownLimitButton.ButtonPushedFcn = createCallbackFcn(app, @SetDownLimitButtonPushed, true);
            app.SetDownLimitButton.Enable = 'off';
            app.SetDownLimitButton.Position = [1333 492 100 23];
            app.SetDownLimitButton.Text = 'Set Down Limit';

            % Create LiquidfrontExtractionButton
            app.LiquidfrontExtractionButton = uibutton(app.TimeDomainTab, 'push');
            app.LiquidfrontExtractionButton.ButtonPushedFcn = createCallbackFcn(app, @LiquidfrontExtractionButtonPushed, true);
            app.LiquidfrontExtractionButton.BackgroundColor = [1 1 1];
            app.LiquidfrontExtractionButton.FontWeight = 'bold';
            app.LiquidfrontExtractionButton.Position = [24 68 193 25];
            app.LiquidfrontExtractionButton.Text = 'Liquidfront Extraction';

            % Create KernelSmoothingPanel
            app.KernelSmoothingPanel = uipanel(app.TimeDomainTab);
            app.KernelSmoothingPanel.Title = 'Kernel Smoothing';
            app.KernelSmoothingPanel.Position = [16 102 216 59];

            % Create ApplyButton
            app.ApplyButton = uibutton(app.KernelSmoothingPanel, 'push');
            app.ApplyButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyButtonPushed, true);
            app.ApplyButton.Position = [115 8 91 23];
            app.ApplyButton.Text = 'Apply';

            % Create gausswinEditFieldLabel
            app.gausswinEditFieldLabel = uilabel(app.KernelSmoothingPanel);
            app.gausswinEditFieldLabel.HorizontalAlignment = 'right';
            app.gausswinEditFieldLabel.Position = [7 9 55 22];
            app.gausswinEditFieldLabel.Text = 'gausswin';

            % Create gausswinEditField
            app.gausswinEditField = uieditfield(app.KernelSmoothingPanel, 'numeric');
            app.gausswinEditField.Limits = [1 100];
            app.gausswinEditField.ValueDisplayFormat = '%.0f';
            app.gausswinEditField.Position = [67 9 34 22];
            app.gausswinEditField.Value = 6;

            % Create LoadRawDataButton
            app.LoadRawDataButton = uibutton(app.TimeDomainTab, 'push');
            app.LoadRawDataButton.ButtonPushedFcn = createCallbackFcn(app, @LoadRawDataButtonPushed, true);
            app.LoadRawDataButton.Position = [294 15 121 23];
            app.LoadRawDataButton.Text = 'Load Raw Data';

            % Create LiquidfrontExtractionTab
            app.LiquidfrontExtractionTab = uitab(app.TabGroup);
            app.LiquidfrontExtractionTab.Title = 'Liquidfront Extraction';

            % Create UIAxesLE1
            app.UIAxesLE1 = uiaxes(app.LiquidfrontExtractionTab);
            title(app.UIAxesLE1, 'Terahertz Reflectometry')
            xlabel(app.UIAxesLE1, 'Time (sec)')
            ylabel(app.UIAxesLE1, 'Displaement (mm)')
            app.UIAxesLE1.FontWeight = 'bold';
            app.UIAxesLE1.Box = 'on';
            app.UIAxesLE1.XGrid = 'on';
            app.UIAxesLE1.YGrid = 'on';
            app.UIAxesLE1.FontSize = 12;
            app.UIAxesLE1.Position = [294 50 650 700];

            % Create UIAxesLE2
            app.UIAxesLE2 = uiaxes(app.LiquidfrontExtractionTab);
            title(app.UIAxesLE2, 'Liquid Front Reflection')
            xlabel(app.UIAxesLE2, 'Time (sec)')
            ylabel(app.UIAxesLE2, 'E filed (a.u.)')
            app.UIAxesLE2.FontWeight = 'bold';
            app.UIAxesLE2.Box = 'on';
            app.UIAxesLE2.XGrid = 'on';
            app.UIAxesLE2.YGrid = 'on';
            app.UIAxesLE2.FontSize = 12;
            app.UIAxesLE2.Position = [960 507 470 240];

            % Create UIAxesLE3
            app.UIAxesLE3 = uiaxes(app.LiquidfrontExtractionTab);
            title(app.UIAxesLE3, 'Liquid Front Ingress')
            xlabel(app.UIAxesLE3, 'Time (sec)')
            ylabel(app.UIAxesLE3, 'Displacement (mm)')
            app.UIAxesLE3.FontWeight = 'bold';
            app.UIAxesLE3.Box = 'on';
            app.UIAxesLE3.XGrid = 'on';
            app.UIAxesLE3.YGrid = 'on';
            app.UIAxesLE3.FontSize = 12;
            app.UIAxesLE3.Position = [960 53 470 427];

            % Create ROISelectionPanel
            app.ROISelectionPanel = uipanel(app.LiquidfrontExtractionTab);
            app.ROISelectionPanel.Title = 'ROI Selection';
            app.ROISelectionPanel.Position = [18 487 264 161];

            % Create AlphaDropDown_2Label
            app.AlphaDropDown_2Label = uilabel(app.ROISelectionPanel);
            app.AlphaDropDown_2Label.HorizontalAlignment = 'right';
            app.AlphaDropDown_2Label.Position = [14 80 35 22];
            app.AlphaDropDown_2Label.Text = 'Alpha';

            % Create AlphaDropDown
            app.AlphaDropDown = uidropdown(app.ROISelectionPanel);
            app.AlphaDropDown.Items = {'1.0', '0.7', '0.5', '0.3', '0.1'};
            app.AlphaDropDown.Position = [53 80 75 22];
            app.AlphaDropDown.Value = '1.0';

            % Create ExtColormapDropDown
            app.ExtColormapDropDown = uidropdown(app.ROISelectionPanel);
            app.ExtColormapDropDown.Items = {'parula', 'jet', 'copper', 'bone', 'hot'};
            app.ExtColormapDropDown.Position = [17 52 111 23];
            app.ExtColormapDropDown.Value = 'parula';

            % Create LfPlotButton
            app.LfPlotButton = uibutton(app.ROISelectionPanel, 'push');
            app.LfPlotButton.ButtonPushedFcn = createCallbackFcn(app, @LfPlotButtonPushed, true);
            app.LfPlotButton.BackgroundColor = [1 1 1];
            app.LfPlotButton.FontWeight = 'bold';
            app.LfPlotButton.Position = [153 56 91 47];
            app.LfPlotButton.Text = 'Replot';

            % Create DrawPolylineButton
            app.DrawPolylineButton = uibutton(app.ROISelectionPanel, 'push');
            app.DrawPolylineButton.ButtonPushedFcn = createCallbackFcn(app, @DrawPolylineButtonPushed, true);
            app.DrawPolylineButton.BackgroundColor = [1 1 1];
            app.DrawPolylineButton.FontWeight = 'bold';
            app.DrawPolylineButton.Position = [153 8 90 41];
            app.DrawPolylineButton.Text = 'Draw Polyline';

            % Create ROIwidthEditFieldLabel
            app.ROIwidthEditFieldLabel = uilabel(app.ROISelectionPanel);
            app.ROIwidthEditFieldLabel.HorizontalAlignment = 'right';
            app.ROIwidthEditFieldLabel.Position = [18 18 58 22];
            app.ROIwidthEditFieldLabel.Text = 'ROI width';

            % Create ROIwidthEditField
            app.ROIwidthEditField = uieditfield(app.ROISelectionPanel, 'numeric');
            app.ROIwidthEditField.Limits = [1 200];
            app.ROIwidthEditField.ValueDisplayFormat = '%.0f';
            app.ROIwidthEditField.Position = [82 18 44 22];
            app.ROIwidthEditField.Value = 30;

            % Create DisplayTabletCentreCheckBox
            app.DisplayTabletCentreCheckBox = uicheckbox(app.ROISelectionPanel);
            app.DisplayTabletCentreCheckBox.Text = 'Display Tablet Centre';
            app.DisplayTabletCentreCheckBox.Position = [19 112 136 22];
            app.DisplayTabletCentreCheckBox.Value = true;

            % Create Upto3mmCheckBox
            app.Upto3mmCheckBox = uicheckbox(app.ROISelectionPanel);
            app.Upto3mmCheckBox.Text = 'Up to 3mm';
            app.Upto3mmCheckBox.Position = [166 112 81 22];
            app.Upto3mmCheckBox.Value = true;

            % Create DataComparisonButton
            app.DataComparisonButton = uibutton(app.LiquidfrontExtractionTab, 'push');
            app.DataComparisonButton.ButtonPushedFcn = createCallbackFcn(app, @DataComparisonButtonPushed, true);
            app.DataComparisonButton.FontWeight = 'bold';
            app.DataComparisonButton.Position = [33 159 246 33];
            app.DataComparisonButton.Text = 'Data Comparison';

            % Create GeneralInformationPanel_LE
            app.GeneralInformationPanel_LE = uipanel(app.LiquidfrontExtractionTab);
            app.GeneralInformationPanel_LE.Title = 'General Information';
            app.GeneralInformationPanel_LE.Position = [18 656 263 84];

            % Create DataLengthEditField_2Label_2
            app.DataLengthEditField_2Label_2 = uilabel(app.GeneralInformationPanel_LE);
            app.DataLengthEditField_2Label_2.Position = [3 34 74 22];
            app.DataLengthEditField_2Label_2.Text = ' Data Length';

            % Create dataLengthEditField_LE
            app.dataLengthEditField_LE = uieditfield(app.GeneralInformationPanel_LE, 'numeric');
            app.dataLengthEditField_LE.ValueDisplayFormat = '%.0f';
            app.dataLengthEditField_LE.Position = [77 35 40 20];

            % Create NumberofScansEditField_2Label_2
            app.NumberofScansEditField_2Label_2 = uilabel(app.GeneralInformationPanel_LE);
            app.NumberofScansEditField_2Label_2.HorizontalAlignment = 'right';
            app.NumberofScansEditField_2Label_2.Position = [24 10 48 22];
            app.NumberofScansEditField_2Label_2.Text = 'Number';

            % Create dataNumberEditField_LE
            app.dataNumberEditField_LE = uieditfield(app.GeneralInformationPanel_LE, 'numeric');
            app.dataNumberEditField_LE.ValueDisplayFormat = '%.0f';
            app.dataNumberEditField_LE.Position = [77 11 40 20];

            % Create ThicknessmmEditField_2Label
            app.ThicknessmmEditField_2Label = uilabel(app.GeneralInformationPanel_LE);
            app.ThicknessmmEditField_2Label.HorizontalAlignment = 'right';
            app.ThicknessmmEditField_2Label.Position = [119 34 90 22];
            app.ThicknessmmEditField_2Label.Text = 'Thickness (mm)';

            % Create ThicknessmmEditField_LE
            app.ThicknessmmEditField_LE = uieditfield(app.GeneralInformationPanel_LE, 'numeric');
            app.ThicknessmmEditField_LE.Limits = [0 Inf];
            app.ThicknessmmEditField_LE.ValueDisplayFormat = '%5.2f';
            app.ThicknessmmEditField_LE.Position = [214 34 40 22];

            % Create n_effLabel
            app.n_effLabel = uilabel(app.GeneralInformationPanel_LE);
            app.n_effLabel.HorizontalAlignment = 'right';
            app.n_effLabel.Position = [119 8 92 22];
            app.n_effLabel.Text = 'Refractive Index';

            % Create RefractiveIndexEditField_LE
            app.RefractiveIndexEditField_LE = uieditfield(app.GeneralInformationPanel_LE, 'numeric');
            app.RefractiveIndexEditField_LE.Limits = [0 Inf];
            app.RefractiveIndexEditField_LE.ValueDisplayFormat = '%5.2f';
            app.RefractiveIndexEditField_LE.Editable = 'off';
            app.RefractiveIndexEditField_LE.Position = [214 8 40 22];

            % Create LinearFittingPanel
            app.LinearFittingPanel = uipanel(app.LiquidfrontExtractionTab);
            app.LinearFittingPanel.Title = 'Linear Fitting';
            app.LinearFittingPanel.Position = [19 281 263 197];

            % Create LiquidIngressTimesecEditFieldLabel
            app.LiquidIngressTimesecEditFieldLabel = uilabel(app.LinearFittingPanel);
            app.LiquidIngressTimesecEditFieldLabel.HorizontalAlignment = 'right';
            app.LiquidIngressTimesecEditFieldLabel.FontWeight = 'bold';
            app.LiquidIngressTimesecEditFieldLabel.Position = [16 13 149 22];
            app.LiquidIngressTimesecEditFieldLabel.Text = 'Liquid Ingress Time (sec)';

            % Create LiquidIngressTimesecEditField
            app.LiquidIngressTimesecEditField = uieditfield(app.LinearFittingPanel, 'numeric');
            app.LiquidIngressTimesecEditField.ValueDisplayFormat = '%5.2f';
            app.LiquidIngressTimesecEditField.Editable = 'off';
            app.LiquidIngressTimesecEditField.FontWeight = 'bold';
            app.LiquidIngressTimesecEditField.Position = [175 13 64 22];

            % Create CaculateFittingParametersButton
            app.CaculateFittingParametersButton = uibutton(app.LinearFittingPanel, 'push');
            app.CaculateFittingParametersButton.ButtonPushedFcn = createCallbackFcn(app, @CaculateFittingParametersButtonPushed, true);
            app.CaculateFittingParametersButton.BackgroundColor = [1 1 1];
            app.CaculateFittingParametersButton.FontWeight = 'bold';
            app.CaculateFittingParametersButton.Position = [16 136 231 28];
            app.CaculateFittingParametersButton.Text = 'Caculate Fitting Parameters';

            % Create kmmsEditFieldLabel
            app.kmmsEditFieldLabel = uilabel(app.LinearFittingPanel);
            app.kmmsEditFieldLabel.HorizontalAlignment = 'right';
            app.kmmsEditFieldLabel.Position = [21 75 52 22];
            app.kmmsEditFieldLabel.Text = 'k (mm/s)';

            % Create kmmsEditField
            app.kmmsEditField = uieditfield(app.LinearFittingPanel, 'numeric');
            app.kmmsEditField.ValueDisplayFormat = '%5.2f';
            app.kmmsEditField.Position = [78 75 50 22];

            % Create dmmEditFieldLabel
            app.dmmEditFieldLabel = uilabel(app.LinearFittingPanel);
            app.dmmEditFieldLabel.HorizontalAlignment = 'right';
            app.dmmEditFieldLabel.Position = [135 75 43 22];
            app.dmmEditFieldLabel.Text = 'd (mm)';

            % Create dmmEditField
            app.dmmEditField = uieditfield(app.LinearFittingPanel, 'numeric');
            app.dmmEditField.ValueDisplayFormat = '%5.2f';
            app.dmmEditField.Position = [189 75 50 22];

            % Create FittingFunctionktdLabel
            app.FittingFunctionktdLabel = uilabel(app.LinearFittingPanel);
            app.FittingFunctionktdLabel.FontSize = 14;
            app.FittingFunctionktdLabel.FontWeight = 'bold';
            app.FittingFunctionktdLabel.Position = [22 104 156 22];
            app.FittingFunctionktdLabel.Text = 'Fitting Function: kt + d';

            % Create R2EditFieldLabel
            app.R2EditFieldLabel = uilabel(app.LinearFittingPanel);
            app.R2EditFieldLabel.HorizontalAlignment = 'right';
            app.R2EditFieldLabel.Position = [37 48 26 22];
            app.R2EditFieldLabel.Text = 'R^2';

            % Create R2EditField
            app.R2EditField = uieditfield(app.LinearFittingPanel, 'numeric');
            app.R2EditField.ValueDisplayFormat = '%5.2f';
            app.R2EditField.Position = [78 48 50 22];

            % Create RMSEEditFieldLabel
            app.RMSEEditFieldLabel = uilabel(app.LinearFittingPanel);
            app.RMSEEditFieldLabel.HorizontalAlignment = 'right';
            app.RMSEEditFieldLabel.Position = [135 48 40 22];
            app.RMSEEditFieldLabel.Text = 'RMSE';

            % Create RMSEEditField
            app.RMSEEditField = uieditfield(app.LinearFittingPanel, 'numeric');
            app.RMSEEditField.ValueDisplayFormat = '%5.2f';
            app.RMSEEditField.Position = [189 48 50 22];

            % Create SampleNameEditFieldLabel
            app.SampleNameEditFieldLabel = uilabel(app.LiquidfrontExtractionTab);
            app.SampleNameEditFieldLabel.HorizontalAlignment = 'right';
            app.SampleNameEditFieldLabel.Position = [25 209 81 22];
            app.SampleNameEditFieldLabel.Text = 'Sample Name';

            % Create SampleNameEditField
            app.SampleNameEditField = uieditfield(app.LiquidfrontExtractionTab, 'text');
            app.SampleNameEditField.Position = [111 209 168 22];

            % Create AutoSaveFiguresCheckBox
            app.AutoSaveFiguresCheckBox = uicheckbox(app.LiquidfrontExtractionTab);
            app.AutoSaveFiguresCheckBox.Text = 'Auto-Save Figures';
            app.AutoSaveFiguresCheckBox.Position = [28 242 121 22];
            app.AutoSaveFiguresCheckBox.Value = true;

            % Create SaveFigureButton_LE
            app.SaveFigureButton_LE = uibutton(app.LiquidfrontExtractionTab, 'push');
            app.SaveFigureButton_LE.ButtonPushedFcn = createCallbackFcn(app, @SaveFigureButton_LEPushed, true);
            app.SaveFigureButton_LE.Position = [159 242 113 23];
            app.SaveFigureButton_LE.Text = 'Save Figure';

            % Create DataComparisonTab
            app.DataComparisonTab = uitab(app.TabGroup);
            app.DataComparisonTab.Title = 'Data Comparison';

            % Create UIAxesCD2
            app.UIAxesCD2 = uiaxes(app.DataComparisonTab);
            title(app.UIAxesCD2, 'Liquid Front Reflection')
            xlabel(app.UIAxesCD2, 'Time (sec)')
            ylabel(app.UIAxesCD2, 'E field intensity (a.u.)')
            app.UIAxesCD2.FontWeight = 'bold';
            app.UIAxesCD2.Box = 'on';
            app.UIAxesCD2.FontSize = 12;
            app.UIAxesCD2.Position = [838 27 600 230];

            % Create UIAxesCD1
            app.UIAxesCD1 = uiaxes(app.DataComparisonTab);
            title(app.UIAxesCD1, 'Liquid Front Ingress')
            xlabel(app.UIAxesCD1, 'Time (sec)')
            ylabel(app.UIAxesCD1, 'Displacement (mm)')
            app.UIAxesCD1.FontWeight = 'bold';
            app.UIAxesCD1.Box = 'on';
            app.UIAxesCD1.FontSize = 12;
            app.UIAxesCD1.Position = [837 262 600 490];

            % Create GroupButton
            app.GroupButton = uibutton(app.DataComparisonTab, 'push');
            app.GroupButton.ButtonPushedFcn = createCallbackFcn(app, @GroupButtonPushed, true);
            app.GroupButton.Position = [236 659 118 30];
            app.GroupButton.Text = 'Group';

            % Create UngroupButton
            app.UngroupButton = uibutton(app.DataComparisonTab, 'push');
            app.UngroupButton.ButtonPushedFcn = createCallbackFcn(app, @UngroupButtonPushed, true);
            app.UngroupButton.Position = [235 621 119 30];
            app.UngroupButton.Text = 'Ungroup';

            % Create RemoveButton_DC2
            app.RemoveButton_DC2 = uibutton(app.DataComparisonTab, 'push');
            app.RemoveButton_DC2.ButtonPushedFcn = createCallbackFcn(app, @RemoveButton_DC2Pushed, true);
            app.RemoveButton_DC2.Position = [498 419 84 30];
            app.RemoveButton_DC2.Text = 'Remove';

            % Create MeasurementListBoxLabel
            app.MeasurementListBoxLabel = uilabel(app.DataComparisonTab);
            app.MeasurementListBoxLabel.HorizontalAlignment = 'right';
            app.MeasurementListBoxLabel.Position = [21 725 79 22];
            app.MeasurementListBoxLabel.Text = 'Measurement';

            % Create MeasurementListBox
            app.MeasurementListBox = uilistbox(app.DataComparisonTab);
            app.MeasurementListBox.Items = {};
            app.MeasurementListBox.Multiselect = 'on';
            app.MeasurementListBox.ValueChangedFcn = createCallbackFcn(app, @MeasurementListBoxValueChanged, true);
            app.MeasurementListBox.Position = [19 414 206 310];
            app.MeasurementListBox.Value = {};

            % Create BatchNameEditFieldLabel
            app.BatchNameEditFieldLabel = uilabel(app.DataComparisonTab);
            app.BatchNameEditFieldLabel.HorizontalAlignment = 'right';
            app.BatchNameEditFieldLabel.Position = [239 722 72 22];
            app.BatchNameEditFieldLabel.Text = 'Batch Name';

            % Create BatchNameEditField
            app.BatchNameEditField = uieditfield(app.DataComparisonTab, 'text');
            app.BatchNameEditField.Position = [237 697 117 25];

            % Create BatchListBoxLabel
            app.BatchListBoxLabel = uilabel(app.DataComparisonTab);
            app.BatchListBoxLabel.HorizontalAlignment = 'right';
            app.BatchListBoxLabel.Position = [368 724 33 22];
            app.BatchListBoxLabel.Text = 'Batch';

            % Create BatchListBox
            app.BatchListBox = uilistbox(app.DataComparisonTab);
            app.BatchListBox.Items = {};
            app.BatchListBox.Multiselect = 'on';
            app.BatchListBox.ValueChangedFcn = createCallbackFcn(app, @BatchListBoxValueChanged, true);
            app.BatchListBox.Position = [366 414 115 309];
            app.BatchListBox.Value = {};

            % Create BatchDetailListBoxLabel
            app.BatchDetailListBoxLabel = uilabel(app.DataComparisonTab);
            app.BatchDetailListBoxLabel.HorizontalAlignment = 'right';
            app.BatchDetailListBoxLabel.Position = [531 725 87 22];
            app.BatchDetailListBoxLabel.Text = 'Batch Contents';

            % Create BatchDetailListBox
            app.BatchDetailListBox = uilistbox(app.DataComparisonTab);
            app.BatchDetailListBox.Items = {};
            app.BatchDetailListBox.Position = [491 457 181 266];
            app.BatchDetailListBox.Value = {};

            % Create RemoveButton_DC1
            app.RemoveButton_DC1 = uibutton(app.DataComparisonTab, 'push');
            app.RemoveButton_DC1.ButtonPushedFcn = createCallbackFcn(app, @RemoveButton_DC1Pushed, true);
            app.RemoveButton_DC1.Position = [238 419 84 30];
            app.RemoveButton_DC1.Text = 'Remove';

            % Create AddButton
            app.AddButton = uibutton(app.DataComparisonTab, 'push');
            app.AddButton.ButtonPushedFcn = createCallbackFcn(app, @AddButtonPushed, true);
            app.AddButton.Position = [235 583 119 30];
            app.AddButton.Text = 'Add';

            % Create AssigndatainworkspaceButton
            app.AssigndatainworkspaceButton = uibutton(app.DataComparisonTab, 'push');
            app.AssigndatainworkspaceButton.ButtonPushedFcn = createCallbackFcn(app, @AssigndatainworkspaceButtonPushed, true);
            app.AssigndatainworkspaceButton.FontWeight = 'bold';
            app.AssigndatainworkspaceButton.Position = [335 11 192 32];
            app.AssigndatainworkspaceButton.Text = 'Assign data in workspace';

            % Create SaveProjectButton
            app.SaveProjectButton = uibutton(app.DataComparisonTab, 'push');
            app.SaveProjectButton.ButtonPushedFcn = createCallbackFcn(app, @SaveProjectButtonPushed, true);
            app.SaveProjectButton.FontWeight = 'bold';
            app.SaveProjectButton.Position = [540 11 120 32];
            app.SaveProjectButton.Text = 'Save Project';

            % Create LoadProjectButton
            app.LoadProjectButton = uibutton(app.DataComparisonTab, 'push');
            app.LoadProjectButton.ButtonPushedFcn = createCallbackFcn(app, @LoadProjectButtonPushed, true);
            app.LoadProjectButton.FontWeight = 'bold';
            app.LoadProjectButton.Position = [672 11 120 32];
            app.LoadProjectButton.Text = 'Load Project';

            % Create UITable
            app.UITable = uitable(app.DataComparisonTab);
            app.UITable.ColumnName = {'Sample Name'; 'Thickness (mm)'; 'n_eff'; 'Ingress Time (s)'; 'k'; 'd'; 'R^2'; 'RMSE'};
            app.UITable.RowName = {};
            app.UITable.ColumnEditable = [true false false false false false false false];
            app.UITable.CellEditCallback = createCallbackFcn(app, @UITableCellEdit, true);
            app.UITable.Position = [19 56 805 326];

            % Create FittingFunctionParametersLabel
            app.FittingFunctionParametersLabel = uilabel(app.DataComparisonTab);
            app.FittingFunctionParametersLabel.FontWeight = 'bold';
            app.FittingFunctionParametersLabel.Position = [27 384 270 22];
            app.FittingFunctionParametersLabel.Text = 'Fitting Function Parameters';

            % Create PlotButtonGroup
            app.PlotButtonGroup = uibuttongroup(app.DataComparisonTab);
            app.PlotButtonGroup.Title = 'Plot';
            app.PlotButtonGroup.Position = [684 640 120 80];

            % Create IndividualButton
            app.IndividualButton = uiradiobutton(app.PlotButtonGroup);
            app.IndividualButton.Text = 'Individual';
            app.IndividualButton.Position = [15 32 73 22];
            app.IndividualButton.Value = true;

            % Create BatchButton
            app.BatchButton = uiradiobutton(app.PlotButtonGroup);
            app.BatchButton.Text = 'Batch';
            app.BatchButton.Position = [15 5 54 22];

            % Create LegendCheckBox
            app.LegendCheckBox = uicheckbox(app.DataComparisonTab);
            app.LegendCheckBox.Text = 'Legend';
            app.LegendCheckBox.Position = [700 615 62 22];
            app.LegendCheckBox.Value = true;

            % Create PlotButton_DC
            app.PlotButton_DC = uibutton(app.DataComparisonTab, 'push');
            app.PlotButton_DC.ButtonPushedFcn = createCallbackFcn(app, @PlotButton_DCPushed, true);
            app.PlotButton_DC.Position = [685 531 120 30];
            app.PlotButton_DC.Text = 'Plot';

            % Create PlotButton_DC3D
            app.PlotButton_DC3D = uibutton(app.DataComparisonTab, 'push');
            app.PlotButton_DC3D.ButtonPushedFcn = createCallbackFcn(app, @PlotButton_DC3DPushed, true);
            app.PlotButton_DC3D.Position = [685 494 120 30];
            app.PlotButton_DC3D.Text = '3D Plot (New)';

            % Create PlotButton_DCNew
            app.PlotButton_DCNew = uibutton(app.DataComparisonTab, 'push');
            app.PlotButton_DCNew.ButtonPushedFcn = createCallbackFcn(app, @PlotButton_DCNewPushed, true);
            app.PlotButton_DCNew.Position = [685 457 120 30];
            app.PlotButton_DCNew.Text = 'Colour Plot (New)';

            % Create ExportTableasCSVFormatButton
            app.ExportTableasCSVFormatButton = uibutton(app.DataComparisonTab, 'push');
            app.ExportTableasCSVFormatButton.ButtonPushedFcn = createCallbackFcn(app, @ExportTableasCSVFormatButtonPushed, true);
            app.ExportTableasCSVFormatButton.FontWeight = 'bold';
            app.ExportTableasCSVFormatButton.Position = [38 11 192 32];
            app.ExportTableasCSVFormatButton.Text = 'Export Table as CSV Format';

            % Create UpButton
            app.UpButton = uibutton(app.DataComparisonTab, 'push');
            app.UpButton.ButtonPushedFcn = createCallbackFcn(app, @UpButtonPushed, true);
            app.UpButton.Position = [238 493 84 30];
            app.UpButton.Text = 'Up';

            % Create DownButton
            app.DownButton = uibutton(app.DataComparisonTab, 'push');
            app.DownButton.ButtonPushedFcn = createCallbackFcn(app, @DownButtonPushed, true);
            app.DownButton.Position = [238 456 84 30];
            app.DownButton.Text = 'Down';

            % Create FrequencyDomainTab
            app.FrequencyDomainTab = uitab(app.TabGroup);
            app.FrequencyDomainTab.Title = 'Frequency Domain';

            % Create UIAxesFD1
            app.UIAxesFD1 = uiaxes(app.FrequencyDomainTab);
            title(app.UIAxesFD1, 'Power spectrum')
            xlabel(app.UIAxesFD1, 'Time/Position (sec/mm)')
            ylabel(app.UIAxesFD1, 'Frequency (THz)')
            app.UIAxesFD1.PlotBoxAspectRatio = [1.0607476635514 1 1];
            app.UIAxesFD1.FontWeight = 'bold';
            app.UIAxesFD1.XTickLabelRotation = 0;
            app.UIAxesFD1.YTickLabelRotation = 0;
            app.UIAxesFD1.ZTickLabelRotation = 0;
            app.UIAxesFD1.Box = 'on';
            app.UIAxesFD1.FontSize = 12;
            app.UIAxesFD1.Position = [262 235 570 520];

            % Create UIAxesFD2
            app.UIAxesFD2 = uiaxes(app.FrequencyDomainTab);
            title(app.UIAxesFD2, 'Phase')
            xlabel(app.UIAxesFD2, 'Time/Position (sec/mm)')
            ylabel(app.UIAxesFD2, 'Frequency (THz)')
            zlabel(app.UIAxesFD2, 'Time (ps)')
            app.UIAxesFD2.PlotBoxAspectRatio = [1.0607476635514 1 1];
            app.UIAxesFD2.FontWeight = 'bold';
            app.UIAxesFD2.ZDir = 'reverse';
            app.UIAxesFD2.XTickLabelRotation = 0;
            app.UIAxesFD2.YTickLabelRotation = 0;
            app.UIAxesFD2.ZTickLabelRotation = 0;
            app.UIAxesFD2.Box = 'on';
            app.UIAxesFD2.FontSize = 12;
            app.UIAxesFD2.Position = [840 235 570 520];

            % Create UIAxesFD3
            app.UIAxesFD3 = uiaxes(app.FrequencyDomainTab);
            xlabel(app.UIAxesFD3, 'Frequency (THz)')
            ylabel(app.UIAxesFD3, 'E field (a.u.)')
            app.UIAxesFD3.PlotBoxAspectRatio = [3.546875 1 1];
            app.UIAxesFD3.FontWeight = 'bold';
            app.UIAxesFD3.Box = 'on';
            app.UIAxesFD3.FontSize = 12;
            app.UIAxesFD3.Position = [263 41 570 180];

            % Create UIAxesFD4
            app.UIAxesFD4 = uiaxes(app.FrequencyDomainTab);
            xlabel(app.UIAxesFD4, 'Frequency (THz)')
            ylabel(app.UIAxesFD4, 'Phase')
            app.UIAxesFD4.PlotBoxAspectRatio = [3.546875 1 1];
            app.UIAxesFD4.FontWeight = 'bold';
            app.UIAxesFD4.Box = 'on';
            app.UIAxesFD4.FontSize = 12;
            app.UIAxesFD4.Position = [841 41 570 180];

            % Create FourierTransformPanel
            app.FourierTransformPanel = uipanel(app.FrequencyDomainTab);
            app.FourierTransformPanel.Title = 'Fourier Transform';
            app.FourierTransformPanel.Position = [13 403 240 290];

            % Create TransformButton
            app.TransformButton = uibutton(app.FourierTransformPanel, 'push');
            app.TransformButton.ButtonPushedFcn = createCallbackFcn(app, @TransformButtonPushed, true);
            app.TransformButton.FontWeight = 'bold';
            app.TransformButton.Position = [26 11 195 25];
            app.TransformButton.Text = 'Transform';

            % Create FrequencyRangeTHzLabel
            app.FrequencyRangeTHzLabel = uilabel(app.FourierTransformPanel);
            app.FrequencyRangeTHzLabel.FontWeight = 'bold';
            app.FrequencyRangeTHzLabel.Position = [12 244 139 22];
            app.FrequencyRangeTHzLabel.Text = 'Frequency Range (THz)';

            % Create fromLabel
            app.fromLabel = uilabel(app.FourierTransformPanel);
            app.fromLabel.HorizontalAlignment = 'right';
            app.fromLabel.Position = [59 220 59 23];
            app.fromLabel.Text = ' from ';

            % Create FromFreqEditField
            app.FromFreqEditField = uieditfield(app.FourierTransformPanel, 'numeric');
            app.FromFreqEditField.Limits = [0 5];
            app.FromFreqEditField.ValueDisplayFormat = '%.1f';
            app.FromFreqEditField.Position = [122 221 40 22];
            app.FromFreqEditField.Value = 0.2;

            % Create toLabel
            app.toLabel = uilabel(app.FourierTransformPanel);
            app.toLabel.HorizontalAlignment = 'right';
            app.toLabel.Position = [158 220 25 23];
            app.toLabel.Text = 'to';

            % Create ToFreqEditField
            app.ToFreqEditField = uieditfield(app.FourierTransformPanel, 'numeric');
            app.ToFreqEditField.Limits = [1 10];
            app.ToFreqEditField.ValueDisplayFormat = '%.1f';
            app.ToFreqEditField.Position = [188 221 40 22];
            app.ToFreqEditField.Value = 3.5;

            % Create UpsamplingLabel
            app.UpsamplingLabel = uilabel(app.FourierTransformPanel);
            app.UpsamplingLabel.FontWeight = 'bold';
            app.UpsamplingLabel.Position = [12 203 74 22];
            app.UpsamplingLabel.Text = 'Upsampling';

            % Create ZeroFillingpowerofSpinnerLabel
            app.ZeroFillingpowerofSpinnerLabel = uilabel(app.FourierTransformPanel);
            app.ZeroFillingpowerofSpinnerLabel.Position = [37 179 129 23];
            app.ZeroFillingpowerofSpinnerLabel.Text = 'Zero Filling, + power of';

            % Create ZeroFillingpowerofSpinner
            app.ZeroFillingpowerofSpinner = uispinner(app.FourierTransformPanel);
            app.ZeroFillingpowerofSpinner.Limits = [0 4];
            app.ZeroFillingpowerofSpinner.Position = [165 180 65 22];
            app.ZeroFillingpowerofSpinner.Value = 1;

            % Create UnwrappingLabel
            app.UnwrappingLabel = uilabel(app.FourierTransformPanel);
            app.UnwrappingLabel.FontWeight = 'bold';
            app.UnwrappingLabel.Position = [13 157 75 22];
            app.UnwrappingLabel.Text = 'Unwrapping';

            % Create StartFrequencyTHzEditFieldLabel
            app.StartFrequencyTHzEditFieldLabel = uilabel(app.FourierTransformPanel);
            app.StartFrequencyTHzEditFieldLabel.HorizontalAlignment = 'right';
            app.StartFrequencyTHzEditFieldLabel.Position = [56 135 124 22];
            app.StartFrequencyTHzEditFieldLabel.Text = 'Start Frequency (THz)';

            % Create StartFrequencyTHzEditField
            app.StartFrequencyTHzEditField = uieditfield(app.FourierTransformPanel, 'numeric');
            app.StartFrequencyTHzEditField.Limits = [0.2 1.5];
            app.StartFrequencyTHzEditField.ValueDisplayFormat = '%5.1f';
            app.StartFrequencyTHzEditField.Position = [185 135 41 22];
            app.StartFrequencyTHzEditField.Value = 0.8;

            % Create ExtrapolationRangeTHzLabel
            app.ExtrapolationRangeTHzLabel = uilabel(app.FourierTransformPanel);
            app.ExtrapolationRangeTHzLabel.FontWeight = 'bold';
            app.ExtrapolationRangeTHzLabel.Position = [14 109 155 22];
            app.ExtrapolationRangeTHzLabel.Text = 'Extrapolation Range (THz)';

            % Create fromLabel_2
            app.fromLabel_2 = uilabel(app.FourierTransformPanel);
            app.fromLabel_2.HorizontalAlignment = 'right';
            app.fromLabel_2.Position = [87 85 29 22];
            app.fromLabel_2.Text = 'from';

            % Create FromEpolFreqEditField
            app.FromEpolFreqEditField = uieditfield(app.FourierTransformPanel, 'numeric');
            app.FromEpolFreqEditField.Limits = [0 5];
            app.FromEpolFreqEditField.ValueDisplayFormat = '%5.2f';
            app.FromEpolFreqEditField.Position = [121 85 40 22];
            app.FromEpolFreqEditField.Value = 0.2;

            % Create toLabel_2
            app.toLabel_2 = uilabel(app.FourierTransformPanel);
            app.toLabel_2.HorizontalAlignment = 'right';
            app.toLabel_2.Position = [157 85 25 22];
            app.toLabel_2.Text = 'to';

            % Create ToEpolFreqEditField
            app.ToEpolFreqEditField = uieditfield(app.FourierTransformPanel, 'numeric');
            app.ToEpolFreqEditField.Limits = [0 5];
            app.ToEpolFreqEditField.ValueDisplayFormat = '%5.2f';
            app.ToEpolFreqEditField.Position = [187 85 40 22];
            app.ToEpolFreqEditField.Value = 0.4;

            % Create FunctionDropDownLabel
            app.FunctionDropDownLabel = uilabel(app.FourierTransformPanel);
            app.FunctionDropDownLabel.FontWeight = 'bold';
            app.FunctionDropDownLabel.Position = [13 48 58 23];
            app.FunctionDropDownLabel.Text = 'Function';

            % Create FunctionDropDown
            app.FunctionDropDown = uidropdown(app.FourierTransformPanel);
            app.FunctionDropDown.Items = {'barthannwin', 'blackman', 'blackmanharris', 'bohmanwin', 'chebwin', 'flattopwin', 'gausswin', 'hamming', 'hann', 'kaiser', 'nuttallwin', 'parzenwin', 'rectwin', 'taylorwin', 'triang', 'tukeywin'};
            app.FunctionDropDown.Position = [80 49 146 22];
            app.FunctionDropDown.Value = 'rectwin';

            % Create DataInformationPanel
            app.DataInformationPanel = uipanel(app.FrequencyDomainTab);
            app.DataInformationPanel.Title = 'Data Information';
            app.DataInformationPanel.Position = [13 700 239 55];

            % Create DataLengthEditField_2Label
            app.DataLengthEditField_2Label = uilabel(app.DataInformationPanel);
            app.DataLengthEditField_2Label.HorizontalAlignment = 'right';
            app.DataLengthEditField_2Label.Position = [7 8 70 22];
            app.DataLengthEditField_2Label.Text = ' DataLength';

            % Create dataLengthEditField_FD
            app.dataLengthEditField_FD = uieditfield(app.DataInformationPanel, 'numeric');
            app.dataLengthEditField_FD.ValueDisplayFormat = '%.0f';
            app.dataLengthEditField_FD.Position = [86 9 40 20];

            % Create NumberofScansEditField_2Label
            app.NumberofScansEditField_2Label = uilabel(app.DataInformationPanel);
            app.NumberofScansEditField_2Label.HorizontalAlignment = 'right';
            app.NumberofScansEditField_2Label.Position = [132 8 48 22];
            app.NumberofScansEditField_2Label.Text = 'Number';

            % Create dataNumberEditField_FD
            app.dataNumberEditField_FD = uieditfield(app.DataInformationPanel, 'numeric');
            app.dataNumberEditField_FD.ValueDisplayFormat = '%.0f';
            app.dataNumberEditField_FD.Position = [190 9 40 20];

            % Create NextButton
            app.NextButton = uibutton(app.FrequencyDomainTab, 'push');
            app.NextButton.ButtonPushedFcn = createCallbackFcn(app, @NextButtonPushed, true);
            app.NextButton.FontWeight = 'bold';
            app.NextButton.Enable = 'off';
            app.NextButton.Position = [27 53 215 24];
            app.NextButton.Text = 'Next';

            % Create SingleMeasurementPanel
            app.SingleMeasurementPanel = uipanel(app.FrequencyDomainTab);
            app.SingleMeasurementPanel.Title = 'Single Measurement';
            app.SingleMeasurementPanel.Position = [13 140 240 108];

            % Create EnableButton_2
            app.EnableButton_2 = uibutton(app.SingleMeasurementPanel, 'state');
            app.EnableButton_2.ValueChangedFcn = createCallbackFcn(app, @EnableButton_2ValueChanged, true);
            app.EnableButton_2.Text = 'Enable';
            app.EnableButton_2.FontWeight = 'bold';
            app.EnableButton_2.Position = [78 56 66 20];

            % Create AutoScanButton
            app.AutoScanButton = uibutton(app.SingleMeasurementPanel, 'push');
            app.AutoScanButton.ButtonPushedFcn = createCallbackFcn(app, @AutoScanButtonPushed, true);
            app.AutoScanButton.Enable = 'off';
            app.AutoScanButton.Position = [151 56 71 21];
            app.AutoScanButton.Text = 'Auto-Scan';

            % Create LocationSliderLabel
            app.LocationSliderLabel = uilabel(app.SingleMeasurementPanel);
            app.LocationSliderLabel.HorizontalAlignment = 'right';
            app.LocationSliderLabel.Enable = 'off';
            app.LocationSliderLabel.Position = [10 56 50 22];
            app.LocationSliderLabel.Text = 'Location';

            % Create LocationSlider
            app.LocationSlider = uislider(app.SingleMeasurementPanel);
            app.LocationSlider.ValueChangedFcn = createCallbackFcn(app, @LocationSliderValueChanged, true);
            app.LocationSlider.ValueChangingFcn = createCallbackFcn(app, @LocationSliderValueChanging, true);
            app.LocationSlider.Enable = 'off';
            app.LocationSlider.Position = [13 39 207 3];

            % Create AssiginFFTDatainworkspaceButton
            app.AssiginFFTDatainworkspaceButton = uibutton(app.FrequencyDomainTab, 'push');
            app.AssiginFFTDatainworkspaceButton.ButtonPushedFcn = createCallbackFcn(app, @AssiginFFTDatainworkspaceButtonPushed, true);
            app.AssiginFFTDatainworkspaceButton.Position = [27 19 215 24];
            app.AssiginFFTDatainworkspaceButton.Text = 'Assigin FFT Data in workspace';

            % Create ColormapcontrolPanel
            app.ColormapcontrolPanel = uipanel(app.FrequencyDomainTab);
            app.ColormapcontrolPanel.Title = 'Colormap control';
            app.ColormapcontrolPanel.Position = [13 255 239 141];

            % Create PlotButton_3
            app.PlotButton_3 = uibutton(app.ColormapcontrolPanel, 'push');
            app.PlotButton_3.ButtonPushedFcn = createCallbackFcn(app, @PlotButton_3Pushed, true);
            app.PlotButton_3.Position = [110 8 117 23];
            app.PlotButton_3.Text = 'Plot';

            % Create DataRangeEditFieldLabel_2
            app.DataRangeEditFieldLabel_2 = uilabel(app.ColormapcontrolPanel);
            app.DataRangeEditFieldLabel_2.HorizontalAlignment = 'right';
            app.DataRangeEditFieldLabel_2.Position = [8 92 69 22];
            app.DataRangeEditFieldLabel_2.Text = 'Data Range';

            % Create DataRangeFromEditField_2
            app.DataRangeFromEditField_2 = uieditfield(app.ColormapcontrolPanel, 'numeric');
            app.DataRangeFromEditField_2.ValueDisplayFormat = '%5.2f';
            app.DataRangeFromEditField_2.Editable = 'off';
            app.DataRangeFromEditField_2.Position = [86 92 43 22];

            % Create Label_3
            app.Label_3 = uilabel(app.ColormapcontrolPanel);
            app.Label_3.HorizontalAlignment = 'right';
            app.Label_3.Position = [144 94 25 22];
            app.Label_3.Text = '-';

            % Create DataRangeToEditField_2
            app.DataRangeToEditField_2 = uieditfield(app.ColormapcontrolPanel, 'numeric');
            app.DataRangeToEditField_2.ValueDisplayFormat = '%5.2f';
            app.DataRangeToEditField_2.Editable = 'off';
            app.DataRangeToEditField_2.Position = [184 94 43 22];

            % Create DOIRangeLabel_2
            app.DOIRangeLabel_2 = uilabel(app.ColormapcontrolPanel);
            app.DOIRangeLabel_2.HorizontalAlignment = 'right';
            app.DOIRangeLabel_2.Position = [9 66 64 22];
            app.DOIRangeLabel_2.Text = 'AOI Range';

            % Create AOIRangeFromEditField_2
            app.AOIRangeFromEditField_2 = uieditfield(app.ColormapcontrolPanel, 'numeric');
            app.AOIRangeFromEditField_2.ValueDisplayFormat = '%5.2f';
            app.AOIRangeFromEditField_2.Position = [86 66 43 22];

            % Create Label_4
            app.Label_4 = uilabel(app.ColormapcontrolPanel);
            app.Label_4.HorizontalAlignment = 'right';
            app.Label_4.Position = [144 66 25 22];
            app.Label_4.Text = '-';

            % Create AOIRangeToEditField_2
            app.AOIRangeToEditField_2 = uieditfield(app.ColormapcontrolPanel, 'numeric');
            app.AOIRangeToEditField_2.ValueDisplayFormat = '%5.2f';
            app.AOIRangeToEditField_2.Position = [184 66 43 22];

            % Create ColormapDropDown_4Label
            app.ColormapDropDown_4Label = uilabel(app.ColormapcontrolPanel);
            app.ColormapDropDown_4Label.Position = [14 37 82 22];
            app.ColormapDropDown_4Label.Text = 'Colormap';

            % Create ftColormapDropDown
            app.ftColormapDropDown = uidropdown(app.ColormapcontrolPanel);
            app.ftColormapDropDown.Items = {'parula', 'jet', 'copper', 'bone', 'hot'};
            app.ftColormapDropDown.Position = [109 36 118 23];
            app.ftColormapDropDown.Value = 'parula';

            % Create ftColorbarCheckBox
            app.ftColorbarCheckBox = uicheckbox(app.ColormapcontrolPanel);
            app.ftColorbarCheckBox.Text = 'Colorbar';
            app.ftColorbarCheckBox.Position = [14 9 69 22];

            % Create StopProcessButton
            app.StopProcessButton = uibutton(app.DipTabUIFigure, 'push');
            app.StopProcessButton.ButtonPushedFcn = createCallbackFcn(app, @StopProcessButtonPushed, true);
            app.StopProcessButton.FontWeight = 'bold';
            app.StopProcessButton.Position = [959 11 110 23];
            app.StopProcessButton.Text = 'Stop Process';

            % Create Image
            app.Image = uiimage(app.DipTabUIFigure);
            app.Image.Position = [19 846 67 60];
            app.Image.ImageSource = fullfile(pathToMLAPP, 'Images', 'dotTHz_logo.png');

            % Create DipTabLabel
            app.DipTabLabel = uilabel(app.DipTabUIFigure);
            app.DipTabLabel.FontSize = 32;
            app.DipTabLabel.FontWeight = 'bold';
            app.DipTabLabel.FontAngle = 'italic';
            app.DipTabLabel.FontColor = [0.149 0.149 0.149];
            app.DipTabLabel.Position = [85 859 228 47];
            app.DipTabLabel.Text = 'DipTab';

            % Create DeployButton
            app.DeployButton = uibutton(app.DipTabUIFigure, 'push');
            app.DeployButton.ButtonPushedFcn = createCallbackFcn(app, @DeployButtonPushed, true);
            app.DeployButton.FontSize = 14;
            app.DeployButton.FontWeight = 'bold';
            app.DeployButton.Position = [886 851 122 28];
            app.DeployButton.Text = 'Deploy';

            % Create DescriptionEditFieldLabel
            app.DescriptionEditFieldLabel = uilabel(app.DipTabUIFigure);
            app.DescriptionEditFieldLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.DescriptionEditFieldLabel.HorizontalAlignment = 'right';
            app.DescriptionEditFieldLabel.Position = [1022 853 66 22];
            app.DescriptionEditFieldLabel.Text = 'Description';

            % Create SampleDescriptionEditField
            app.SampleDescriptionEditField = uieditfield(app.DipTabUIFigure, 'text');
            app.SampleDescriptionEditField.HorizontalAlignment = 'right';
            app.SampleDescriptionEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.SampleDescriptionEditField.Position = [1098 854 303 20];

            % Show the figure after all components are created
            app.DipTabUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = DipTab_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.DipTabUIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.DipTabUIFigure)
        end
    end
end