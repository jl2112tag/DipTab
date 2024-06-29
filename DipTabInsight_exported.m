classdef DipTabInsight_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        DipTabInsightUIFigure           matlab.ui.Figure
        SampleDescriptionEditField      matlab.ui.control.EditField
        DescriptionEditFieldLabel       matlab.ui.control.Label
        DeployButton                    matlab.ui.control.Button
        DipTabInsightLabel              matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        StopProcessButton               matlab.ui.control.Button
        TabGroup                        matlab.ui.container.TabGroup
        TimedomainTab                   matlab.ui.container.Tab
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
        NextButton_2                    matlab.ui.control.Button
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
        SpectrumPanel                   matlab.ui.container.Panel
        AutoScanSpectrumCheckBox        matlab.ui.control.CheckBox
        PowerSpectrumButton             matlab.ui.control.Button
        SpectrogramButton               matlab.ui.control.Button
        ColormapPanel                   matlab.ui.container.Panel
        DCheckBox                       matlab.ui.control.CheckBox
        colorbarCheckBox                matlab.ui.control.CheckBox
        PlotButton                      matlab.ui.control.Button
        ColormapDropDown                matlab.ui.control.DropDown
        GuideLinesPanel                 matlab.ui.container.Panel
        SetInitButton                   matlab.ui.control.Button
        EnableButton                    matlab.ui.control.StateButton
        GeneralInformationPanel         matlab.ui.container.Panel
        TimeSpacingsEditField           matlab.ui.control.NumericEditField
        TimeSpacingsEditFieldLabel      matlab.ui.control.Label
        ToFSpacingpsEditField           matlab.ui.control.NumericEditField
        XSpacingpsEditFieldLabel        matlab.ui.control.Label
        DataNumberEditField             matlab.ui.control.NumericEditField
        DataNumberLabel                 matlab.ui.control.Label
        DataLengthEditField             matlab.ui.control.NumericEditField
        DataLengthLabel                 matlab.ui.control.Label
        UIAxes14                        matlab.ui.control.UIAxes
        UIAxes13                        matlab.ui.control.UIAxes
        UIAxes12                        matlab.ui.control.UIAxes
        UIAxes11                        matlab.ui.control.UIAxes
        FrequencydomainTab              matlab.ui.container.Tab
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
        dataNumberEditField_2           matlab.ui.control.NumericEditField
        NumberofScansEditField_2Label   matlab.ui.control.Label
        dataLengthEditField_2           matlab.ui.control.NumericEditField
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
        UIAxes24                        matlab.ui.control.UIAxes
        UIAxes23                        matlab.ui.control.UIAxes
        UIAxes22                        matlab.ui.control.UIAxes
        UIAxes21                        matlab.ui.control.UIAxes
        ExtractionTab                   matlab.ui.container.Tab
        BatchManagementButton           matlab.ui.control.Button
        ExtractReflectionPointsPanel    matlab.ui.container.Panel
        ROIDropDown                     matlab.ui.control.DropDown
        ROIDropDownLabel                matlab.ui.control.Label
        FREEHANDROIButton               matlab.ui.control.Button
        ExcludeuptopsEditField          matlab.ui.control.NumericEditField
        ExcludeuptopsEditFieldLabel     matlab.ui.control.Label
        ROITheasholdSlider              matlab.ui.control.Slider
        ROITheasholdSliderLabel         matlab.ui.control.Label
        ExcludeLowerReflectionsCheckBox  matlab.ui.control.CheckBox
        LiquidIngressTimesecEditField   matlab.ui.control.NumericEditField
        LiquidIngressTimesecEditFieldLabel  matlab.ui.control.Label
        Range220EditField               matlab.ui.control.NumericEditField
        Range220EditFieldLabel          matlab.ui.control.Label
        DBSCANNeighborhoodRadiusSwitch  matlab.ui.control.Switch
        DBSCANNeighborhoodRadiusSwitchLabel  matlab.ui.control.Label
        DistancetoCentrepsEditField     matlab.ui.control.NumericEditField
        DistancetoCentrepsEditFieldLabel  matlab.ui.control.Label
        ALGORITHMROIButton              matlab.ui.control.Button
        RefractiveIndexEditField_T3     matlab.ui.control.NumericEditField
        n_effLabel                      matlab.ui.control.Label
        CaculateIngressTimeButton       matlab.ui.control.Button
        SampleDescriptionEditField_T3   matlab.ui.control.EditField
        SampleNameEditField_T3Label     matlab.ui.control.Label
        ExtractLiquidFrontButton        matlab.ui.control.Button
        LfPlotButton                    matlab.ui.control.Button
        ExtColormapDropDown             matlab.ui.control.DropDown
        CmapLabel                       matlab.ui.control.Label
        AlphaDropDown                   matlab.ui.control.DropDown
        AlphaDropDown_2Label            matlab.ui.control.Label
        UIAxes35                        matlab.ui.control.UIAxes
        UIAxes34                        matlab.ui.control.UIAxes
        UIAxes33                        matlab.ui.control.UIAxes
        UIAxes32                        matlab.ui.control.UIAxes
        UIAxes31                        matlab.ui.control.UIAxes
        BatchAnalysisTab                matlab.ui.container.Tab
        DatasettoUseButtonGroup         matlab.ui.container.ButtonGroup
        MeanButton                      matlab.ui.control.RadioButton
        ScatteredButton                 matlab.ui.control.RadioButton
        ColourPlotNewButton             matlab.ui.control.Button
        DPlotNewButton                  matlab.ui.control.Button
        PlotButton_2                    matlab.ui.control.Button
        FittingCheckBox                 matlab.ui.control.CheckBox
        LegendCheckBox                  matlab.ui.control.CheckBox
        StyleButtonGroup                matlab.ui.container.ButtonGroup
        SDErrorBarButton                matlab.ui.control.RadioButton
        SDShadowedButton                matlab.ui.control.RadioButton
        AllRangeButton                  matlab.ui.control.RadioButton
        DisplayButtonGroup              matlab.ui.container.ButtonGroup
        BatchButton                     matlab.ui.control.RadioButton
        IndividualButton                matlab.ui.control.RadioButton
        FittingFunctionParametersunderdevelopmentLabel  matlab.ui.control.Label
        UITable                         matlab.ui.control.Table
        LoadProjectButton               matlab.ui.control.Button
        SaveProjectButton               matlab.ui.control.Button
        AssigndatainworkspaceButton     matlab.ui.control.Button
        AddButton                       matlab.ui.control.Button
        RemoveButton_2                  matlab.ui.control.Button
        BatchDetailListBox              matlab.ui.control.ListBox
        BatchDetailListBoxLabel         matlab.ui.control.Label
        BatchListBox                    matlab.ui.control.ListBox
        BatchListBoxLabel               matlab.ui.control.Label
        BatchNameEditField              matlab.ui.control.EditField
        BatchNameEditFieldLabel         matlab.ui.control.Label
        MeasurementListBox              matlab.ui.control.ListBox
        MeasurementListBoxLabel         matlab.ui.control.Label
        InformationPanel                matlab.ui.container.Panel
        BIBatchEditField                matlab.ui.control.EditField
        BatchLabel                      matlab.ui.control.Label
        BIMeanSpeedEditField            matlab.ui.control.NumericEditField
        MeanSpeedmmminLabel             matlab.ui.control.Label
        BIIngressTimesecEditField       matlab.ui.control.NumericEditField
        IngressTimesecEditFieldLabel    matlab.ui.control.Label
        BIRefractiveIndexEditField      matlab.ui.control.NumericEditField
        RefractiveIndexEditField_2Label  matlab.ui.control.Label
        BIDescriptionEditField          matlab.ui.control.EditField
        SampleDescriptionLabel          matlab.ui.control.Label
        BIDistanceToCentreEditField     matlab.ui.control.NumericEditField
        DistancetoCentremmLabel         matlab.ui.control.Label
        RemoveButton                    matlab.ui.control.Button
        UngroupButton                   matlab.ui.control.Button
        GroupButton                     matlab.ui.control.Button
        UIAxes41                        matlab.ui.control.UIAxes
        UIAxes42                        matlab.ui.control.UIAxes
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
        LfData % Liquid Front Relevant Datasets 
        stopProcess % variable for forcing process stopped     
        Handler % axis component handler
        BData % Batch data
        plotUpdate % display update option
        maxSam % max sample value
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
        
        function tdPlot(app,autoStr)
            samData = app.TData.samData;
            ToF = app.TData.ToF;
            xData = app.TData.xData;
            app.maxSam = max(samData,[],"all");
            
            ax1 = app.UIAxes11;
            ax3 = app.UIAxes13;
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
            
            if app.DCheckBox.Value
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
            ax2 = app.UIAxes12;

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
            app.LfData = [];
            app.Handler = [];
        end
        
        
        function plotSpectra(app)       
            ax = app.UIAxes21;
            
            clo(ax)
            clo(app.UIAxes22)
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
            
            ax = app.UIAxes22;
            app.stopProcess = 0;
            app.EnableButton.Value = false;
            
            clo(ax)
            clo(app.UIAxes23)
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
            ax3 = app.UIAxes23;
            ax4 = app.UIAxes24;
            
            try 
                xData = app.FData.xData;
                freq = app.FData.freq;
            catch ME
                fig = app.DipTabInsightUIFigure;
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

            ax3 = app.UIAxes23;
            ax4 = app.UIAxes24;

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
        
        function LfPlot(app)
            lfRawData = app.LfData.lfRawData;
            lfFtdData = app.LfData.lfFtdData;
            lfTime = app.LfData.lfTime;
            lfDis = app.LfData.lfDis;
            alp = str2num(app.AlphaDropDown.Value);
            cmap = app.ExtColormapDropDown.Value;
            normOpt = 0; %normalise two sets of data
            
            if normOpt
                lfRawData = lfRawData/max(lfRawData,[],'all');
                lfFtdData = lfFtdData/max(lfFtdData,[],'all');
            end
           
            ax1 = app.UIAxes31;
            ax2 = app.UIAxes32;
            ax3 = app.UIAxes33;
            ax4 = app.UIAxes34;
            ax5 = app.UIAxes35;
            
            cla(ax1);
            cla(ax2);
            cla(ax3);
            cla(ax4);
            cla(ax5);
            
            imagesc(ax1,lfTime,lfDis,lfRawData,'AlphaData',alp);
            imagesc(ax2,lfTime,lfDis,lfFtdData,'AlphaData',alp);
            
            ax1.YDir = 'normal';
            ax2.YDir = 'normal';
            
            axis(ax1,'tight');
            axis(ax2,'tight');
            
            colormap(ax1,cmap);
            colormap(ax2,cmap);

            app.LiquidIngressTimesecEditField.Value = 0;
            
        end
        
        
        function updateBatchList(app)
            % batcht list update
            bNum = size(app.BData.batch,2);
            ListBoxItems={};
            cnt = 1;
            
            for bIdx = 1:bNum
                AddItem = app.BData.batch{bIdx};
                if ~isempty(AddItem)&&~sum(strcmp(AddItem,ListBoxItems))
                   ListBoxItems(cnt) = {AddItem};
                   cnt = cnt+1;
                end
            end
            
            app.BatchListBox.Items = ListBoxItems;
            app.BatchListBox.ItemsData = (1:length(ListBoxItems));           
        end
        
        function plotIndividual(app)
            linearFitExtract = 1; % option boolean for extracting linear fitting function from the measurement data
            mItems = app.MeasurementListBox.Value;
            listItems = app.MeasurementListBox.Items;
            dataset = app.DatasettoUseButtonGroup.SelectedObject.Text;
            
            if isempty(mItems)
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            dpMat = app.BData.dpMat;
            dpMatMean = app.BData.dpMatMean;
            
            ax1 = app.UIAxes41;
            ax2 = app.UIAxes42;
            axis(ax1,'tight');
            axis(ax2,'tight');
            xlim(ax2,[0 inf]);
            ylim(ax2,[0 inf]);
            cla(ax1);
            cla(ax2);
            hold(ax1,'on');
            hold(ax2,'on');
            
            % fit function data for table fill-in
            ks1 = zeros(size(mItems,2),1);
            frss1 = zeros(size(mItems,2),1);
            frms1 = zeros(size(mItems,2),1);
            ks2 = zeros(size(mItems,2),1);
            ds2 = zeros(size(mItems,2),1);
            frss2 = zeros(size(mItems,2),1);
            frms2 = zeros(size(mItems,2),1);
            
            xData = cell(size(mItems,2));
            fData = cell(size(mItems,2));
            
            lh1 = [];
            lh2 = [];
            cnt = 1;
            
            for idx = mItems
                if isequal(dataset,'Scattered')
                    xData = dpMat{idx}(:,1);    
                    yData_Intensity = dpMat{idx}(:,3);
                    yData_Displacement = dpMat{idx}(:,2);
                else
                    xData = dpMatMean{idx}(:,1);    
                    yData_Intensity = dpMatMean{idx}(:,3);
                    yData_Displacement = dpMatMean{idx}(:,2);
                end

                h1 = plot(ax1,xData,yData_Intensity,'.');
                h2 = plot(ax2,xData,yData_Displacement,'.');
                lh1(cnt) = h1;
                lh2(cnt) = h2;
                
%                 if app.FittingCheckBox.Value;
%                     [k1,frs1,frm1,fitData1,k2,d,frs2,frm2,fitData2] = PLFitting(app,idx,'individual');
%                     ks1(cnt) = k1;
%                     frss1(cnt) = frs1; %rsquare
%                     frms1(cnt) = frm1; %RMSE
%                     h2 = plot(ax,dpMat{idx}(1,1:length(fitData1)),fitData1,'--');
%                     h2.Color = [h1.Color, 0.4];
%                     ks2(cnt) = k2;
%                     ds2(cnt) = d;
%                     frss2(cnt) = frs2; %rsquare
%                     frms2(cnt) = frm2; %RMSE
%                     h3 = plot(ax,dpMat{idx}(1,length(fitData1):end),fitData2,'--');
%                     h3.Color = [h1.Color, 0.4];
%                 end
%                 
%                 if linearFitExtract
%                     app.BData.dpMat{idx}(3,:) = dpMat{idx}(2,:)-(k2*dpMat{idx}(1,:)+d);    
%                 end
                
                cnt = cnt + 1;
            end
            
            if app.FittingCheckBox.Value;
                T = table(listItems([mItems])',ks1,frss1,frms1,ks2,ds2,frss2,frms2);
                app.UITable.Data = T;    
            end
            
            
            
            if app.LegendCheckBox.Value
                legend(ax1,lh1,listItems([mItems]),"Location","southeast","Interpreter","none");
                legend(ax2,lh2,listItems([mItems]),"Location","southeast","Interpreter","none");
            else
                legend(ax1,"off");
                legend(ax2,"off");
            end
        end
        
        function plotBatch(app)
            bItems = app.BatchListBox.Value;
            sBatch = app.BatchListBox.Items(bItems);
            
            if isempty(bItems)
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'No batch selected','Warning');
                return;
            end
            
            Data = app.BData;
            dpMat = Data.dpMat;
            batch = Data.batch;
            meta = Data.meta;
            tNum = size(dpMat,2);
            bMat = {}; %for batch data set plotting
            
            ax = app.UIAxes42;
            axis(ax,'tight');
            xlim(ax,[0, inf]);
            ylim(ax,[0, inf]);
            cla(ax);
            hold(ax,'on');
            cnt = 1;
            
            for bIdx = sBatch % sBatch: selected batch(es)
                bIdxMat = strcmp(batch,bIdx);
                bMat(cnt,1:2) = {cnt,bIdx};
                bNum = sum(bIdxMat);
                cnt2 = 0;
                itpItvs = zeros(sum(bIdxMat),1); % interpolation interval vectors
                mEndTs = zeros(sum(bIdxMat),1); % measurement end time
                phaseTransitionTimes = zeros(sum(bIdxMat),1); % phase transition time
                
                %calculate a suitable interpolation interval for batch data
                for idx = 1:tNum
                    if bIdxMat(idx)
                        itpItvs(idx) = meta(idx).MeasurementInterval;
                        phaseTransitionTimes(idx) = meta(idx).PhaseTransitionTime;
                        mEndTs(idx) = dpMat{idx}(1,end);
                    end
                end
                
                itpItv = min(nonzeros(itpItvs));
                mEndT = min(nonzeros(mEndTs));
                phaseTransitionTime = mean(phaseTransitionTimes);
                
                for idx = 1:tNum
                    if bIdxMat(idx)
                        cnt2 = cnt2 + 1;
                        ubLoc = sum(dpMat{idx}(1,:) <= mEndT);
                        iT = dpMat{idx}(1,1:ubLoc); %ingress time
                        dT = dpMat{idx}(2,1:ubLoc); %displacement time
                        xq = 0:itpItv:mEndT;
                        vq = interp1(iT,dT,xq,'linear');
                        vq(isnan(vq)) = 0;
                        bMat(cnt,4+cnt2) = {[xq;vq]};   
                    end
                end
                
                ptNum = size(xq,2);
                bcMat= zeros(cnt2,ptNum); %batch commmon data
                
                for idx = 1:cnt2
                    bcMat(idx,:) = bMat{cnt,4+idx}(2,:);
                end
                
                %liquid front batch data
                LFb = zeros(5,ptNum);
                LFb(1,:) = (0:1:ptNum-1)*itpItv; %ingress time
                LFb(2,:) = min(bcMat,[],1); % min values
                LFb(3,:) = max(bcMat,[],1); % max values
                LFb(4,:) = mean(bcMat); % mean values
                LFb(5,:) = std(bcMat); % standard deviations
                

                bMat(cnt,4) = {LFb};
                bMat(cnt,3) = {phaseTransitionTime};
                cnt = cnt + 1;
       end

            
            app.BData.bMat = bMat;
            
            
            pMethod = app.StyleButtonGroup.SelectedObject.Text;
            cnt = 1;
            lh = []; %line handler
            ks1 = zeros(size(bItems,2),1);
            frss1 = zeros(size(bItems,2),1);
            frms1 = zeros(size(bItems,2),1);
            ks2 = zeros(size(bItems,2),1);
            ds2 = zeros(size(bItems,2),1);
            frss2 = zeros(size(bItems,2),1);
            frms2 = zeros(size(bItems,2),1);
            xData = cell(size(bItems,2));
            fData = cell(size(bItems,2));
            
            for bIdx = sBatch
                sbMat = bMat{cnt,4}; %selected batch matrix
                
                switch pMethod
                    case 'All Range'
                        h1 = plot(ax,sbMat(1,:),sbMat(4,:));
                        lh(cnt) = h1;
                        xBound = [sbMat(1,:),flip(sbMat(1,:))];
                        yBound = [sbMat(2,:),flip(sbMat(3,:))];
                        patch(ax,xBound,yBound,h1.Color,'EdgeColor','none','FaceAlpha',0.1);
                    case 'SD (Shadowed)'
                        h1 = plot(ax,sbMat(1,:),sbMat(4,:));
                        lh(cnt) = h1;
                        xBound = [sbMat(1,:),flip(sbMat(1,:))];
                        yBound = [sbMat(4,:)-sbMat(5,:),flip(sbMat(4,:))+flip(sbMat(5,:))];
                        patch(ax,xBound,yBound,h1.Color,'EdgeColor','none','FaceAlpha',0.1);
                    case 'SD (ErrorBar)'
                        h1 = errorbar(ax,sbMat(1,:),sbMat(4,:),sbMat(5,:));
                end
                
                if app.FittingCheckBox.Value;
                    [k1,frs1,frm1,fitData1,k2,d,frs2,frm2,fitData2] = PLFitting(app,cnt,'batch');
                    ks1(cnt) = k1;
                    frss1(cnt) = frs1; %rsquare
                    frms1(cnt) = frm1; %RMSE
                    h2 = plot(ax,sbMat(1,1:length(fitData1)),fitData1,'--');
                    h2.Color = [h1.Color, 0.4];
                    ks2(cnt) = k2;
                    ds2(cnt) = d;
                    frss2(cnt) = frs2; %rsquare
                    frms2(cnt) = frm2; %RMSE
                    h3 = plot(ax,sbMat(1,length(fitData1):end),fitData2,'--');
                    h3.Color = [h1.Color, 0.4];
                end
                cnt = cnt + 1;
            end
                                    
            if app.LegendCheckBox.Value
                legend(ax,lh,sBatch,"Location","southeast","Interpreter","none");
            end
            
            if app.FittingCheckBox.Value;
                T = table(sBatch',ks1,frss1,frms1,ks2,ds2,frss2,frms2);          
                app.UITable.Data = T;
            end
         
        end
        
        function [k1,frs1,frm1,fitData1,k2,d,frs2,frm2,fitData2] = PLFitting(app,idx,opt)
            
            if isequal(opt,'individual')
                dpMat = app.BData.dpMat{idx};
                meta = app.BData.meta(idx);
                phaseTransitionTime = meta.PhaseTransitionTime;
                xData = dpMat(1,:)';
                yData = dpMat(2,:)';
            else
                bMat = app.BData.bMat{idx,4};
                phaseTransitionTime = app.BData.bMat{idx,3};
                xData = bMat(1,:)';
                yData = bMat(4,:)';   
            end
            
            phaseTransitionLoc = sum(xData <= phaseTransitionTime);
            
            % Phase 2 fitting
            ft = fittype('k2*x + d');
            options = fitoptions('Method','NonlinearLeastSquares','Lower',[0 0 0],'Upper',[100 20 10]);
            [f2, fitness2] = fit(xData(phaseTransitionLoc:end),yData(phaseTransitionLoc:end),ft,options);
            k2 = f2.k2;
            d = f2.d;
            frs2 = fitness2.rsquare;
            frm2 = fitness2.rmse;
            fitData2 = feval(f2,xData(phaseTransitionLoc:end));
            
            % Phase 1 fitting
            ft = fittype('k1*x');
            options = fitoptions('Method','NonlinearLeastSquares','Lower',[0],'Upper',[100]);
            [f1, fitness1] = fit([0, phaseTransitionTime]',[0, fitData2(1)]',ft,options);
            k1 = f1.k1;
            frs1 = fitness1.rsquare;
            frm1 = fitness1.rmse;
            fitData1 = feval(f1,xData(1:phaseTransitionLoc));
        end
          
        
        
        
        function bwDataset(app)
            ax = app.UIAxes35;
            cmap = app.ExtColormapDropDown.Value;
            alp = str2num(app.AlphaDropDown.Value);
            app.SystemStatusEditField.Value = "Dochotomising in process";
            drawnow
            
            try
                lfFtdData = app.LfData.lfFtdData;
                lfDis = app.LfData.lfDis;
                lfTime = app.LfData.lfTime;
            catch ME
            end
            
            threshold = app.ROITheasholdSlider.Value;
            lfBWData = (lfFtdData>=threshold);

            lb = app.ExcludeuptopsEditField.Value; % lower boundary

            if lb>0
                lbLoc = sum(lfDis<lb);
                lfBWData(:,1:lbLoc) = false;
            end

            for idx = 1:length(lfDis)
                if isequal(lfBWData(idx,1),true)
                    break;
                else
                    lfBWData(idx,1) = true;
                end
            end

            imagesc(ax,lfTime,lfDis,lfBWData,"AlphaData",0.1);
            colormap(ax,cmap);
            axis(ax,'tight');
            ax.YDir = "normal";
            app.LfData.lfBWData = lfBWData;
            app.LfData.algoROI = lfBWData;
            app.ROIDropDown.Value = "ALGORITHM ONLY";
            app.SystemStatusEditField.Value = "Dichotomising finished";            
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
            figure(app.DipTabInsightUIFigure);            
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
            app.BData = BData;
            app.MeasurementListBox.Items = ItemList;
            app.MeasurementListBox.ItemsData = (1:length(ItemList));
            
            updateBatchList(app);
            app.SystemStatusEditField.Value = 'Project loaded.';
        end

        % Button pushed function: PlotButton
        function PlotButtonPushed(app, event)
            tdPlot(app);        
        end

        % Value changed function: ColormapDropDown
        function ColormapDropDownValueChanged(app, event)
            value = app.ColormapDropDown.Value;
            tdPlot(app);
        end

        % Value changed function: EnableButton
        function EnableButtonValueChanged(app, event)
            value = app.EnableButton.Value;
            ax1 = app.UIAxes11;
            ax2 = app.UIAxes12;
            ax3 = app.UIAxes13;
            
            try 
                xData = app.TData.xData;
                ToF = app.TData.ToF;
            catch ME
                fig = app.DipTabInsightUIFigure;
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
                app.SetInitButton.Enable = true;
                
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
                app.SetInitButton.Enable = false;
                app.tXPickSlider.Enable = false;
                app.tXPickEditField.Enable = false;
                app.tYPickSlider.Enable = false;
                app.tYPickEditField.Enable = false;
                app.SetDownLimitButton.Enable = false;
                app.SetLeftLimitButton.Enable = false;
                tdPlot(app);
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
            ax = app.UIAxes11;
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
            xData = app.TData.xData; % x-axis
            ToF = app.TData.ToF; % time of flight (ps)
            
            app.DeployButton.Enable = false; % prevent performing twice
            app.SystemStatusEditField.Value = 'Truncating....';
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
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Incorrect X-truncation setting','Warning');
                app.DeployButton.Enable = true;
                app.TruncateButton.Enable = true;
                app.SystemStatusEditField.Value = 'NEXT cancelled';
                return;
            end

           % lb, ub availability check
            if ylbLoc==0 || yubLoc>dataLength || ylbLoc>yubLoc
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Incorrect Y-truncation setting','Warning');
                app.DeployButton.Enable = true;
                app.TruncateButton.Enable = true;
                app.SystemStatusEditField.Value = 'NEXT cancelled';
                return;
            end

            samData = samData(ylbLoc:yubLoc, xlbLoc:xubLoc);
            
            [cmin cmax] = bounds(samData,"all");

            cmin = round(cmin*10^2)*10^-2;
            cmax = round(cmax*10^2)*10^-2;

            app.DataRangeFromEditField.Value = cmin;
            app.DataRangeToEditField.Value = cmax;
            app.AOIRangeFromEditField.Value = cmin;
            app.AOIRangeToEditField.Value = cmax;
                        
            % Scan information panel display
            app.dataLengthEditField_2.Value = dataLength;
            
            % assign truncated Y,X time
            xData = xData(xlbLoc:xubLoc) - xData(xlbLoc);
            app.TData.xData = xData;
            ToF = ToF(ylbLoc:yubLoc) - ToF(ylbLoc);
            app.TData.ToF = ToF;
            
            app.SystemStatusEditField.Value = 'Done';
            app.TData.samData = samData;
            app.TData.ftdSam = samData;
            app.DeployButton.Enable = true;
            app.LeftEditField.Value = 0;
            app.RightEditField.Value = 0;
            app.UpEditField.Value = 0;
            app.DownEditField.Value = 0;
            drawnow
            tdPlot(app);
        end

        % Button pushed function: LfPlotButton
        function LfPlotButtonPushed(app, event)
            LfPlot(app);
            app.SystemStatusEditField.Value = 'Replot Finished';
        end

        % Callback function
        function SaveFigureBRButtonPushed(app, event)
            ax = app.UIAxes22;
            filter = {'*.pdf';'*.*'};
            [filename, filepath] = uiputfile(filter);
            
            if isequal(filename,0)||isequal(filepath,0)
                return;          
            end
            
            fullfile = strcat(filepath,filename);
            exportgraphics(ax,fullfile);
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
            
            BData = app.BData;
            ItemList = app.MeasurementListBox.Items;
            
            
            save(fullfile,'BData','ItemList');
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
            ax = app.UIAxes14;

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
            ax4 = app.UIAxes14;
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
            ax1 = app.UIAxes21;
            ax2 = app.UIAxes22;
            ax3 = app.UIAxes23;
            ax4 = app.UIAxes24;
            
            try 
                xData = app.FData.xData;
                freq = app.FData.freq;
                magData = app.FData.magData;
                phsData = app.FData.phsData;
            catch ME
                fig = app.DipTabInsightUIFigure;
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

        % Callback function
        function tPickSlider_YValueChanging(app, event)
            changingValue = event.Value;
            app.tPickEditField_Y.Value = changingValue;
            
            if app.EnableButton_2.Value
                posT2_YLine(app);
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

        % Callback function
        function FindBottomSurfacePeakButtonPushed(app, event)
            
            try 
                ftdData = app.TData.ftdSam;
                xData = app.TData.xData;
                ingressDepth = app.TData.ToF;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Load Data first','Warning');
                app.EnableButton.Value = false;
                return;
            end
            
            yPick = app.tPickSlider_Y.Value;
            xPick = app.tPickSlider_XTime.Value;
            ax4 = app.UIAxes24;
            cla(ax4);
            
            yloc = sum(xData<=yPick);
            xloc = sum(ingressDepth<=xPick);
            if yloc <= 3
                yloc = 5;
                app.tPickSlider_Y.Value = xData(yloc);
                app.tPickEditField_Y.Value = xData(yloc);
            end
            app.TData.yPick = yloc;
            app.TData.xPick = xloc;
            samMean = mean(ftdData(1:yloc,:));

            try
                [minVal minLoc] = min(samMean(xloc-80:xloc+80));
            catch ME
                [minVal minLoc] = min(samMean(xloc-80:end));
            end
            minLoc = minLoc + xloc - 80 -1;

            app.Handler.ax4plot.yData = samMean;

            plot(ax4,ingressDepth,samMean);
            hold(ax4,'on')
            plot(ax4,ingressDepth(minLoc),minVal,'r*');
            title(ax4,'Bottom Surface Reflection (*)');
            app.Handler.xline_24 = xline(ax4,ingressDepth(minLoc),'r--','LineWidth',1);
            app.TData.btmDis = ingressDepth(minLoc);
%             [pkVal, pkLoc] = max(samMean(minLoc-100:minLoc));
%             plot(ax3,xData(pkLoc+minLoc-100),pkVal,"b*");
%             app.TData.contactPeakGap = xData(minLoc) - xData(minLoc-100+pkLoc);
            
            app.tPickSlider_XTime.Value = ingressDepth(minLoc);
            app.tPickEditField_X.Value = ingressDepth(minLoc);
            posT2_XLine(app);
            
            msgTxt = strcat("Bottom surface reflection peak: ",compose('%5.2f',ingressDepth(minLoc))," mm");
            app.SystemStatusEditField.Value = msgTxt;
            
        end

        % Callback function
        function FindLiquidContactTimeButtonPushed(app, event)
            
            try 
                xData = app.TData.xData;
                xData = app.TData.ToF;
                ftdSam = app.TData.ftdSam;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Load Data first','Warning');
                app.EnableButton.Value = false;
                return;
            end
            
            ax1 = app.UIAxes21;
            ax2 = app.UIAxes22;
            ax3 = app.UIAxes23;
            
            % Liquid contact time (hydration start)
            yPick = app.tPickSlider_Y.Value;
            xPick = app.tPickSlider_XTime.Value;
            yLoc1 = sum(xData <= (yPick - 10))+1; % scan +/- 5 second range for the given y-point
            yLoc2 = sum(xData <= (yPick + 10));
            xLoc = sum(xData <= xPick);
            
            bottomPeaks = ftdSam(yLoc1:yLoc2,xLoc)';
            measureTime = xData(yLoc1:yLoc2);
            
            plot(ax3,measureTime,bottomPeaks);
            axis(ax3,'tight');
            xlabel(ax3,'Time (sec)');
            app.tPickSlider_Xtime.Enable = "on";
            app.tPickSlider_Xtime.Limits = [measureTime(1) measureTime(end)];
            %title(ax3,'Bottom Surface Reflection Peak');
            
            [minVal, minLoc] = max(bottomPeaks);
            liquidContactTime = measureTime(minLoc);
            app.TData.liquidContactTime = liquidContactTime;
            app.Handler.xline_23 = xline(ax3,liquidContactTime,'r--');
            app.tPickSlider_Xtime.Value = liquidContactTime;
            legendText = strcat('Contact Time:',compose("%5.2f",liquidContactTime));
            legend(ax3,app.Handler.xline_23,legendText);
            
            app.Handler.yline_21cTime.Visible = true;
            app.Handler.yline_22cTime.Visible = true;
            
            app.Handler.yline_21cTime.Value = liquidContactTime;
            app.Handler.yline_22cTime.Value = liquidContactTime;
            
        end

        % Button pushed function: NextButton
        function NextButtonPushed(app, event)
            try
                lfRawData = app.LfData.lfRawData;
                lfFtdData = app.LfData.lfFtdData;
                lfTime = app.LfData.lfTime;
                lfDis = app.LfData.lfDis;
                app.LfData.algoROI = [];
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                return;
            end

            LfPlot(app);

            bwLim = [min(lfFtdData,[],"all"), max(lfFtdData,[],"all")];
            app.ROITheasholdSlider.Limits = bwLim;
            app.ROITheasholdSlider.Value = mean(bwLim);
            bwDataset(app);
          
            app.TabGroup.SelectedTab = app.TabGroup.Children(3);
            
        end

        % Button pushed function: ExtractLiquidFrontButton
        function ExtractLiquidFrontButtonPushed(app, event)
            app.stopProcess = 0;

            try
                %lfRawData = app.LfData.lfRawData;
                lfFtdData = app.LfData.lfFtdData;
                lfTime = app.LfData.lfTime;
                lfDis = app.LfData.lfDis;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                return
            end

            ROI = app.ROIDropDown.Value;

            switch ROI
                case "ALGORITHM ONLY"
                    try
                       ROI = app.LfData.algoROI;
                    catch ME
                       fig = app.DipTabInsightUIFigure;
                       uialert(fig,'Algorithm dataset is not ready','Warning');
                       app.EnableButton.Value = false;
                       return;
                    end
                case "FREEHAND ONLY"
                    try
                       ROI = app.LfData.fhROI;
                    catch ME
                       fig = app.DipTabInsightUIFigure;
                       uialert(fig,'Freehand dataset is not ready','Warning');
                       app.EnableButton.Value = false;
                       return;
                    end
                otherwise
                    try
                       algoROI = app.LfData.algoROI;
                       fhROI = app.LfData.fhROI;
                       ROI = algoROI.*fhROI;
                    catch ME
                       fig = app.DipTabInsightUIFigure;
                       uialert(fig,'ROI datasets are not ready','Warning');
                       return;
                    end
            end

            ROI = ROI';
            lfFtdData = lfFtdData';

            measNum = length(lfTime);            
            centreDis = app.DistancetoCentrepsEditField.Value;
            lfMat = []; % liquid front matrix
            
            ax1 = app.UIAxes31;
            ax2 = app.UIAxes32;
            ax3 = app.UIAxes33;
            ax4 = app.UIAxes34;
            
            hold(ax1,'on');
            hold(ax2,'on');
            hold(ax4,"on");
            cla(ax3)
            
                        
            % extract liquid front (LF)
            crtLfLoc = 1; % the current liquid front location in the waveform vector
            %lb = crtLfLoc;
            
            for idx = 1:measNum
                tempVec = diff(ROI(idx,:));
                tempVec = round([ROI(idx,1) tempVec]);
                lbs = find(tempVec == 1);
                ubs = find(tempVec == -1);
                cnt = 1;

                if app.stopProcess
                    app.SystemStatusEditField.Value = "Process aborted";
                    app.SpectrogramButton.Enable = true;
                    return
                end

                for lb = lbs

                    try
                        ub = ubs(cnt);
                    catch ME
                        break
                    end
                    
                    lfVec = lfFtdData(idx,lb:ub);% truncated liquid front vector
                    
                    try
                        %[pks locs] = findpeaks(lfVec);
                        [pks locs] = max(lfVec);
                    catch ME
                        [pks locs] = max(lfVec);
                    end

                    cnt2 = 1;
                    
                    for idx2 = locs
                        lfMat= [lfMat; lfTime(idx), lfDis(idx2+lb-1),pks(cnt2)];
                        cnt2 = cnt2 + 1;
                    end
                    cnt = cnt + 1;
                end                
                progressP = idx/measNum*100;
                progressP = num2str(progressP,'%.0f');
                progressP = strcat("Liquid Front Extracting: ", progressP,"%");
                app.SystemStatusEditField.Value = progressP;
                drawnow
            end

%             assignin("base","lfMat",lfMat);
            
            plot(ax2,lfMat(:,1),lfMat(:,2),'.');
            plot(ax3,lfMat(:,1),lfMat(:,3),'.');
            axis(ax3,'tight');
                   
            % display displacements        
            app.Handler.LFline = plot(ax4,lfMat(:,1),lfMat(:,2),'.');
            axis(ax4,'tight');
            xlim(ax4,[0 inf]);
            ylim(ax4,[0 inf]);
                    
            app.LfData.lfMat = lfMat;
            assignin("base","lfMat",lfMat);
        end

        % Button pushed function: BatchManagementButton
        function BatchManagementButtonPushed(app, event)
           
            try
                lfMat = app.LfData.lfMat;
                lfTime = app.LfData.lfTime;
                ingressTime = app.LiquidIngressTimesecEditField.Value;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                return;
            end
                        
            meta.SampleName = app.SampleDescriptionEditField_T3.Value;
            meta.totalDepth = app.DistancetoCentrepsEditField.Value;
            meta.RefractiveIndex = app.RefractiveIndexEditField_T1.Value;
            meta.IngressTime = ingressTime;
            batch = {''};

            ubLoc = sum(lfMat(:,1)<=ingressTime);
            lfMat(ubLoc:end,:) = [];
            
            try
                dpIdx = size(app.BData.dpMat,2)+1;
            catch ME
                app.BData.dpMat = {};
                dpIdx = 1;
            end
            
            app.BData.dpMat(dpIdx) = {lfMat};
            app.BData.meta(dpIdx) = meta;
            app.BData.batch(dpIdx) = batch;

            % add mean point matrix (scattered points --> mean points)
            meanMat = [];

            for idx = 1:size(lfMat,1)-1
                t = lfMat(idx,1);
                diffVector = lfMat(:,1);
                diffVector = diff(diffVector);
                diffVector(diffVector>0) = 1;
                diffVector = [1; diffVector];
        
                if isequal(diffVector(idx),1)
                    intensity = lfMat(idx,3);
                    displacement = lfMat(idx,2);
                    cnt = 1;
        
                    if isequal(diffVector(idx+1),1)
                        meanMat = [meanMat ; t,displacement,intensity];
                    end
                else
                    intensity = intensity + lfMat(idx,3);
                    displacement = displacement + lfMat(idx,2);
                    cnt = cnt + 1;
        
                    if isequal(diffVector(idx+1),1)
                        meanMat = [meanMat; t, displacement/cnt,intensity/cnt];
                    end
                end        
            end

            app.BData.dpMatMean(dpIdx) = {meanMat};
            
            %measurment list update
            ListBoxItems={};
            
            for MeasNum = 1:dpIdx
                AddItem = app.BData.meta(MeasNum).SampleName;
                ListBoxItems{MeasNum} = AddItem;
            end
            
%             ListBoxItems = app.TD_data.sampleNameList;
            app.MeasurementListBox.ItemsData = (1:MeasNum);
            app.MeasurementListBox.Items = ListBoxItems;
            
            app.TabGroup.SelectedTab = app.TabGroup.Children(4);           
        end

        % Button pushed function: RemoveButton_2
        function RemoveButton_2Pushed(app, event)
            delItem = app.MeasurementListBox.Value;
            ListBoxItems = app.MeasurementListBox.Items;
            
            if isempty(delItem)
                return;
            end
            
            app.BData.dpMat(delItem) = [];
            app.BData.dpMatMean(delItem) = [];
            app.BData.meta(delItem) = [];
            app.BData.batch(delItem) = [];
            ListBoxItems(delItem) = [];

            app.MeasurementListBox.Items = ListBoxItems;
            app.MeasurementListBox.ItemsData = (1:length(ListBoxItems));
            
            MeasurementListBoxValueChanged(app);
        end

        % Button pushed function: AssigndatainworkspaceButton
        function AssigndatainworkspaceButtonPushed(app, event)
              assignin('base',"BData",app.BData);
        end

        % Value changed function: MeasurementListBox
        function MeasurementListBoxValueChanged(app, event)
            value = app.MeasurementListBox.Value;
            
            if isequal(size(value,2),1)
                meta = app.BData.meta(value);
                batch = app.BData.batch{value};

                app.BIBatchEditField.Value = batch;
                app.BIDescriptionEditField.Value = meta.SampleName;
                app.BIDistanceToCentreEditField.Value = meta.totalDepth;
                app.BIRefractiveIndexEditField.Value = meta.RefractiveIndex;
                app.BIIngressTimesecEditField.Value = meta.IngressTime;
                app.BIMeanSpeedEditField.Value = (meta.totalDepth)/(meta.IngressTime)*60;
            else
                app.BIBatchEditField.Value = '';
                app.BIDescriptionEditField.Value = '';
                app.BIDistanceToCentreEditField.Value = 0;
                app.BIRefractiveIndexEditField.Value = 0;
                app.BIIngressTimesecEditField.Value = 0;
                app.BIMeanSpeedEditField.Value = 0;
            end 
        end

        % Button pushed function: GroupButton
        function GroupButtonPushed(app, event)
            bItem = app.MeasurementListBox.Value;          
            bName = {app.BatchNameEditField.Value};
            
            if isempty(bItem)||isequal(bName,'')
                return;
            end
            
            app.BData.batch(bItem) = bName;           
            updateBatchList(app);
            
        end

        % Value changed function: BatchListBox
        function BatchListBoxValueChanged(app, event)
            value = app.BatchListBox.Value;
            sBatch = app.BatchListBox.Items(value);            
            tNum = size(app.BData.batch,2); % data set number
            ListBoxItems={};
            cnt = 1;
            
            for idx = 1:tNum
                meta = app.BData.meta(idx);
                AddItem = {meta.SampleName};
                cBatch = app.BData.batch(idx);
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
            
            app.BData.batch(delItem) = {''};
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
                app.BData.batch(bItem) = bName;
            else
                return;
            end
            
            MeasurementListBoxValueChanged(app);
            BatchListBoxValueChanged(app);
            updateBatchList(app);
            
        end

        % Button pushed function: RemoveButton
        function RemoveButtonPushed(app, event)
            delItem = app.BatchDetailListBox.Value;
            ListBoxItems = app.BatchDetailListBox.Items;
            
            if isempty(delItem)
                return;
            end
            
            tNum = size(app.BData.batch,2); % data set number
            
            for idx = 1:tNum
                meta = app.BData.meta(idx);
                cMeas = {meta.Description};
                sMeas = ListBoxItems(delItem);
                if sum(strcmp(cMeas,sMeas))
                   app.BData.batch{idx}='';
                end
            end
            
            MeasurementListBoxValueChanged(app);
            BatchListBoxValueChanged(app);
           
        end

        % Button pushed function: PlotButton_2
        function PlotButton_2Pushed(app, event)
            plotTarget = app.DisplayButtonGroup.SelectedObject.Text;
            app.UITable.Data = [];
            
            if isequal(plotTarget,'Individual')
                plotIndividual(app);
            else
                plotBatch(app);
            end
        end

        % Value changed function: BIDescriptionEditField
        function BIDescriptionEditFieldValueChanged(app, event)
            value = app.BIDescriptionEditField.Value;
            
        end

        % Button pushed function: CaculateIngressTimeButton
        function CaculateIngressTimeButtonPushed(app, event)
            lfMat = app.LfData.lfMat; % liquid front matrx: each row has [time, distance(ps), peak intensity]
            lfDis = app.LfData.lfDis;
            totNum = size(lfMat,1);
            lfMat = lfMat(floor(4*totNum/10):floor(9.5*totNum/10),:); % 40% to 95%
            centreDis = app.DistancetoCentrepsEditField.Value;
            ax = app.UIAxes34;
            hold(ax,"on");
            
            % Regime 2 fitting (f = k*t^m + 0.4)
            ft = fittype('k*x+d');
            options = fitoptions('Method','NonlinearLeastSquares','Lower',[0 0 0],'Upper',[100 20 10]);
            [f, fitness] = fit(lfMat(:,1),(lfMat(:,2)),ft,options);
            k = f.k;
            d = f.d;
            frs = fitness.rsquare;
            frm = fitness.rmse;

            if app.ExcludeLowerReflectionsCheckBox.Value
                cnt = 0;
                exclfMat = lfMat;
                for idx=1:size(lfMat,1)
                    if lfMat(idx,2) < feval(f,lfMat(idx,1))
                        exclfMat(idx-cnt,:)=[];
                        cnt = cnt+1;
                    end
                end
                lfMat = exclfMat;

                ft = fittype('k*x+d');
                options = fitoptions('Method','NonlinearLeastSquares','Lower',[0 0 0],'Upper',[100 20 10]);
                [f, fitness] = fit(lfMat(:,1),(lfMat(:,2)),ft,options);
                k = f.k;
                d = f.d;
                frs = fitness.rsquare;
                frm = fitness.rmse;
            end
                
            fitData = feval(f,lfMat(:,1)');
            plot(ax,lfMat(:,1)',fitData,'m--');
            ax.YLim = [0 lfDis(end)];
            %txtMsg = strcat("k=",compose('%5.2f',k),", m=", compose('%5.2f',m),", d=", compose('%5.2f',d));
            legend(ax,"Liquid Front","Ingress Time Extrapolation Function","Location","southeast")
            %app.Handler.yline_34.Value = d;
            
            % Calculate the liquid ingress time in seconds
            liquidIngressTime = (centreDis-d)/k;
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
                fig = app.DipTabInsightUIFigure;
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

        % Button pushed function: ALGORITHMROIButton
        function ALGORITHMROIButtonPushed(app, event)
            app.stopProcess = 0;
            ax = app.UIAxes35;
            cmap = app.ExtColormapDropDown.Value;
            alp = str2num(app.AlphaDropDown.Value);
            app.SystemStatusEditField.Value = "Candidate area narrowing...";
            drawnow
            
            
            try
                lfRawData = app.LfData.lfRawData;
                lfBWData = app.LfData.lfBWData;
                lfTime = app.LfData.lfTime;
                lfDis = app.LfData.lfDis;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                app.EnableButton.Value = false;
                return
            end

            autoRadius = app.DBSCANNeighborhoodRadiusSwitch.Value;

            if isequal(autoRadius,"Auto")
                nRadius = [5 10 15];
            else
                nRadius = app.Range220EditField.Value; % DBSCAN neighborhood radius
            end

            algoROI = zeros(size(lfBWData));

            for nR = nRadius
                lfBWData = app.LfData.lfBWData;
                lfBWData = int8(lfBWData);
                lfBWData = imgaussfilt(lfBWData,nR,"Padding","replicate");
                algoROITemp = zeros(size(lfBWData));
            
                % desity clustering
                ptX = 1;
                ptY = 1;
                ptXMax = length(lfDis);
                ptYMax = length(lfTime)-nR+1;
                initPtX = ptX;

                while ptY<ptYMax       
                    for ptX = initPtX
                        xloof = true;
                        while xloof
                            if (ptX+nR-1)>=ptXMax
                                xloof = false;
                            else
                                candArea = lfBWData(ptY:ptY+nR-1,ptX:ptX+nR-1);                            
                                if isequal(sum(candArea(1,:)),0)
                                    xloof = false;
                                else
                                    algoROITemp(ptY:ptY+nR-1,ptX:ptX+nR-1) = candArea;
                                    ptX = ptX + nR;
                                end
                           end                        
                        end
                    end

                    ptY = ptY + 1;
                    initPtX = diff(algoROITemp(ptY,:));
                    initPtX = [algoROITemp(ptY,1), initPtX];
                    initPtX(initPtX < 0) = 0;
                    initPtX = round(initPtX);
                    initPtX = find(initPtX==1);
                    xloof = true;
                 
                    if app.stopProcess
                        app.SystemStatusEditField.Value = "Process aborted";
                        app.SpectrogramButton.Enable = true;
                        return
                    end
                end
                algoROI = algoROI + algoROITemp;
                algoROI(algoROI > 0) = 1;
            end
                        
            app.LfData.algoROI = algoROI;
            imagesc(ax,lfTime,lfDis,algoROI,"AlphaData",alp);
            colormap(ax,cmap);
            axis(ax,'tight');
            ax.YDir = "normal";
            app.SystemStatusEditField.Value = "Candidate area extracted";
        end

        % Value changed function: DBSCANNeighborhoodRadiusSwitch
        function DBSCANNeighborhoodRadiusSwitchValueChanged(app, event)
            value = app.DBSCANNeighborhoodRadiusSwitch.Value;
            if isequal(value,"Manual")
                app.Range220EditField.Editable = "on";
            else
                app.Range220EditField.Editable = "off";
            end
        end

        % Callback function
        function tPickSlider_XtimeValueChanging(app, event)
            changingValue = event.Value;
            liquidContactTime = changingValue;
            ax = app.UIAxes23;
            if app.EnableButton_2.Value
                app.TData.liquidContactTime = liquidContactTime;
                app.Handler.xline_23.Value = liquidContactTime;
                legendText = strcat('Contact Time:',compose("%5.2f",liquidContactTime));
                legend(ax,app.Handler.xline_23,legendText);
            
                app.Handler.yline_21cTime.Visible = true;
                app.Handler.yline_22cTime.Visible = true;
            
                app.Handler.yline_21cTime.Value = liquidContactTime;
                app.Handler.yline_22cTime.Value = liquidContactTime;
            end
        end

        % Value changed function: ROITheasholdSlider
        function ROITheasholdSliderValueChanged(app, event)
            value = app.ROITheasholdSlider.Value;
            bwDataset(app);
        end

        % Value changed function: ExcludeuptopsEditField
        function ExcludeuptopsEditFieldValueChanged(app, event)
            value = app.ExcludeuptopsEditField.Value;
            bwDataset(app);
        end

        % Button pushed function: FREEHANDROIButton
        function FREEHANDROIButtonPushed(app, event)
            try
                lfFtdData = app.LfData.lfFtdData;
                lfTime = app.LfData.lfTime;
                lfDis = app.LfData.lfDis;
            catch ME
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'Dataset is not ready','Warning');
                app.EnableButton.Value = false;
                return
            end
            
            cmap = app.ExtColormapDropDown.Value;
            alp = str2num(app.AlphaDropDown.Value);
            
            ax1 = app.UIAxes32;
            ax2 = app.UIAxes35;
            
            app.SystemStatusEditField.Value = "Please select a freehand region of interest (ROI) at Filtered Measurement AX";

            roi = drawfreehand(ax1,"Color","r");
            fhROI = createMask(roi);
            imagesc(ax2,lfTime,lfDis,fhROI,"AlphaData",alp);
            app.LfData.fhROI = fhROI;
            colormap(ax2,cmap);
            axis(ax2,'tight');
            ax2.YDir = "normal";
            app.ROIDropDown.Value = "FREEHAND ONLY";
            app.SystemStatusEditField.Value = "Freehand ROI is selected.";
        end

        % Value changed function: ROIDropDown
        function ROIDropDownValueChanged(app, event)
            value = app.ROIDropDown.Value;

            if isequal(value,"ALGORITHM + FREEHAND")
                try
                    algoROI = app.LfData.algoROI;
                    fhROI = app.LfData.fhROI;
                    ROI = algoROI.*fhROI;
                catch ME
                    fig = app.DipTabInsightUIFigure;
                    uialert(fig,'ROI datasets are not ready','Warning');
                    return;
                end
                
                ax = app.UIAxes35;
                lfTime = app.LfData.lfTime;
                lfDis = app.LfData.lfDis;
                cmap = app.ExtColormapDropDown.Value;
                alp = str2num(app.AlphaDropDown.Value);
                imagesc(ax,lfDis,lfTime,ROI,"AlphaData",alp);
                colormap(ax,cmap);
                axis(ax,'tight');
                ax.YDir = "normal";
            end
            
        end

        % Button pushed function: DPlotNewButton
        function DPlotNewButtonPushed(app, event)
            % Create UIFigure and hide until all components are created
            fig = figure('Visible', 'on');
            fig.Position = [100 100 1200 800];
            fig.Name = "CaTTrans 3D Plot";

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
            mItems = app.MeasurementListBox.Value;
            listItems = app.MeasurementListBox.Items;
            
            if isempty(mItems)
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            dpMat = app.BData.dpMat;            
            lh1 = [];
            lh = [];
            cnt = 1;
            
            for idx = mItems
                h = plot3(ax,dpMat{idx}(:,1),dpMat{idx}(:,2),dpMat{idx}(:,3),'.');
                lh(cnt) = h;
                
                cnt = cnt + 1;
            end
                        
            if app.LegendCheckBox.Value
                legend(ax,lh,listItems([mItems]),"Location","southeast","Interpreter","none");
            else
                legend(ax,"off");
            end
        end

        % Button pushed function: ColourPlotNewButton
        function ColourPlotNewButtonPushed(app, event)
            % Create UIFigure and hide until all components are created
            fig = uifigure('Visible', 'on');
            fig.Position = [100 100 1200 900];
            fig.Name = "CaTTrans Colour Plot";

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
            mItems = app.MeasurementListBox.Value;
            listItems = app.MeasurementListBox.Items;
            
            if isempty(mItems)
                fig = app.DipTabInsightUIFigure;
                uialert(fig,'No profiles selected','Warning');
                return;                
            end

            dpMat = app.BData.dpMat;            
            lh1 = [];
            lh = [];
            cnt = 1;
            
            for idx = mItems
                h = scatter(ax,dpMat{idx}(:,1),dpMat{idx}(:,2),[],dpMat{idx}(:,3),'.');
                lh(cnt) = h;
                
                cnt = cnt + 1;
            end
                        
            if app.LegendCheckBox.Value
                legend(ax,lh,listItems([mItems]),"Location","southeast","Interpreter","none");
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
            ax = app.UIAxes11;
                        
            if isempty(fullfile)
                     return;
            end

            question = "Select x-axis unit";
            xUnit = questdlg('Select x-axis unit','X-axis Unit','Time (sec)','Position (mm)','Time (sec)');

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

            if isequal(instrument,"<missing>")
                instrument = '';
            end

            % x, y axis units
            xDataItv = timeDiff/measNum;
            app.TData.xData = xDataItv*(0:measNum-1);
            ToF = xSpacing*(0:dataLength-1);
            app.TData.ToF = ToF;
            samData = zeros(dataLength,measNum);

            % Scan information panel display
            app.DataLengthEditField.Value = dataLength;
            app.DataNumberEditField.Value = measNum;
            app.ToFSpacingpsEditField.Value = xSpacing;
            app.TimeSpacingsEditField.Value = xDataItv;
            % app.InstrumentEditField.Value = instrument;
            % app.UserEditField.Value = user;
            app.SampleDescriptionEditField.Value = description;
                                    
            % measurement dataset extraction
            for idx = 1:measNum
                dn = strcat(ListItems{idx},'/ds1');
                measData = h5read(fullfile,dn);

                if peakOption == "Negative"
                    samData(:,idx) = measData(2,:)'*-1;
                else
                    samData(:,idx) = measData(2,:)';
                end                
               
                progressP = idx/measNum*100;
                progressP = num2str(progressP,'%.0f');
                progressP = strcat("Loading: ", progressP,"%");
                app.SystemStatusEditField.Value = progressP;
                drawnow
            end

            app.TData.samData = samData;
            app.SystemStatusEditField.Value = 'Done';
            drawnow

            [cmin cmax] = bounds(samData,"all");

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
            
            tdPlot(app); % painting 2D image display

            %app.DeployButton.Enable = true;
            app.EnableButton.Value = false;
            EnableButtonValueChanged(app);
            app.TruncateButton.Enable = true;
            app.TabGroup.SelectedTab = app.TabGroup.Children(1);
        end

        % Button pushed function: NextButton_2
        function NextButton_2Pushed(app, event)
            samData = app.TData.samData;
            dataLength=size(samData,1);
            measNum = size(samData,2);
            app.dataLengthEditField_2.Value = dataLength;
            app.dataNumberEditField_2.Value = measNum;
            app.DeployButton.Enable = true;
            app.TabGroup.SelectedTab = app.TabGroup.Children(2);
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
            tdPlot(app);
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

        % Button pushed function: SetInitButton
        function SetInitButtonPushed(app, event)
            if app.EnableButton.Value
                initY = app.tXPickSlider.Value;
                initX = app.tYPickSlider.Value;
                app.LeftEditField.Value = initY;
                app.DownEditField.Value = initX;
            else
                return;
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

        % Callback function
        function PlotButton_TDPushed(app, event)
            tdPlot(app);
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
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create DipTabInsightUIFigure and hide until all components are created
            app.DipTabInsightUIFigure = uifigure('Visible', 'off');
            app.DipTabInsightUIFigure.Position = [100 100 1484 916];
            app.DipTabInsightUIFigure.Name = 'DipTab Insight';
            app.DipTabInsightUIFigure.Icon = fullfile(pathToMLAPP, 'Images', 'icon.png');

            % Create ImportthzFileButton
            app.ImportthzFileButton = uibutton(app.DipTabInsightUIFigure, 'push');
            app.ImportthzFileButton.ButtonPushedFcn = createCallbackFcn(app, @ImportthzFileButtonPushed, true);
            app.ImportthzFileButton.FontWeight = 'bold';
            app.ImportthzFileButton.Position = [334 850 114 28];
            app.ImportthzFileButton.Text = 'Import .thz File';

            % Create TerahertzLqiuidFrontDateAnalyserLabel
            app.TerahertzLqiuidFrontDateAnalyserLabel = uilabel(app.DipTabInsightUIFigure);
            app.TerahertzLqiuidFrontDateAnalyserLabel.FontWeight = 'bold';
            app.TerahertzLqiuidFrontDateAnalyserLabel.FontAngle = 'italic';
            app.TerahertzLqiuidFrontDateAnalyserLabel.Position = [84 840 224 29];
            app.TerahertzLqiuidFrontDateAnalyserLabel.Text = 'Terahertz LqiuidFront Date Analyser';

            % Create ProjectNameEditField
            app.ProjectNameEditField = uieditfield(app.DipTabInsightUIFigure, 'text');
            app.ProjectNameEditField.Editable = 'off';
            app.ProjectNameEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.ProjectNameEditField.Position = [463 853 432 22];

            % Create SystemStatusEditFieldLabel
            app.SystemStatusEditFieldLabel = uilabel(app.DipTabInsightUIFigure);
            app.SystemStatusEditFieldLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.SystemStatusEditFieldLabel.HorizontalAlignment = 'right';
            app.SystemStatusEditFieldLabel.Position = [30 12 83 22];
            app.SystemStatusEditFieldLabel.Text = 'System Status';

            % Create SystemStatusEditField
            app.SystemStatusEditField = uieditfield(app.DipTabInsightUIFigure, 'text');
            app.SystemStatusEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.SystemStatusEditField.Position = [128 12 822 22];

            % Create TabGroup
            app.TabGroup = uitabgroup(app.DipTabInsightUIFigure);
            app.TabGroup.AutoResizeChildren = 'off';
            app.TabGroup.Position = [19 46 1446 790];

            % Create TimedomainTab
            app.TimedomainTab = uitab(app.TabGroup);
            app.TimedomainTab.AutoResizeChildren = 'off';
            app.TimedomainTab.Title = 'Time domain';

            % Create UIAxes11
            app.UIAxes11 = uiaxes(app.TimedomainTab);
            title(app.UIAxes11, '2D Image of Terahertz Reflectometry')
            xlabel(app.UIAxes11, 'Time/Position (sec/mm)')
            ylabel(app.UIAxes11, 'Time of flight (ps)')
            app.UIAxes11.PlotBoxAspectRatio = [1 1.13357400722022 1];
            app.UIAxes11.XTickLabelRotation = 0;
            app.UIAxes11.YTickLabelRotation = 0;
            app.UIAxes11.ZTickLabelRotation = 0;
            app.UIAxes11.Box = 'on';
            app.UIAxes11.FontSize = 11;
            app.UIAxes11.Position = [238 53 636 700];

            % Create UIAxes12
            app.UIAxes12 = uiaxes(app.TimedomainTab);
            title(app.UIAxes12, 'Single E-field')
            xlabel(app.UIAxes12, 'Time of flight (ps)')
            ylabel(app.UIAxes12, 'E field (a.u.)')
            app.UIAxes12.Box = 'on';
            app.UIAxes12.FontSize = 11;
            app.UIAxes12.Position = [881 539 560 220];

            % Create UIAxes13
            app.UIAxes13 = uiaxes(app.TimedomainTab);
            title(app.UIAxes13, 'Cummurative E-field')
            xlabel(app.UIAxes13, 'Time/Position (sec/mm)')
            ylabel(app.UIAxes13, 'E field (a.u.)')
            app.UIAxes13.Box = 'on';
            app.UIAxes13.FontSize = 11;
            app.UIAxes13.Position = [883 262 560 220];

            % Create UIAxes14
            app.UIAxes14 = uiaxes(app.TimedomainTab);
            title(app.UIAxes14, 'Spectrogram / Spectrum')
            xlabel(app.UIAxes14, 'Time (ps) / Frequency (THz)')
            ylabel(app.UIAxes14, 'Frequency (THz) / E field')
            zlabel(app.UIAxes14, 'Z')
            app.UIAxes14.Box = 'on';
            app.UIAxes14.FontSize = 11;
            app.UIAxes14.Position = [883 6 560 200];

            % Create GeneralInformationPanel
            app.GeneralInformationPanel = uipanel(app.TimedomainTab);
            app.GeneralInformationPanel.AutoResizeChildren = 'off';
            app.GeneralInformationPanel.Title = 'General Information';
            app.GeneralInformationPanel.Position = [15 633 216 110];

            % Create DataLengthLabel
            app.DataLengthLabel = uilabel(app.GeneralInformationPanel);
            app.DataLengthLabel.HorizontalAlignment = 'right';
            app.DataLengthLabel.Position = [3 61 70 22];
            app.DataLengthLabel.Text = 'Data Length';

            % Create DataLengthEditField
            app.DataLengthEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.DataLengthEditField.ValueDisplayFormat = '%.0f';
            app.DataLengthEditField.Position = [77 61 41 22];

            % Create DataNumberLabel
            app.DataNumberLabel = uilabel(app.GeneralInformationPanel);
            app.DataNumberLabel.HorizontalAlignment = 'right';
            app.DataNumberLabel.Position = [119 61 48 22];
            app.DataNumberLabel.Text = 'Number';

            % Create DataNumberEditField
            app.DataNumberEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.DataNumberEditField.ValueDisplayFormat = '%.0f';
            app.DataNumberEditField.Position = [169 61 41 22];

            % Create XSpacingpsEditFieldLabel
            app.XSpacingpsEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.XSpacingpsEditFieldLabel.Position = [7 34 99 22];
            app.XSpacingpsEditFieldLabel.Text = 'ToF Spacing (ps) ';

            % Create ToFSpacingpsEditField
            app.ToFSpacingpsEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.ToFSpacingpsEditField.ValueDisplayFormat = '%5.3f';
            app.ToFSpacingpsEditField.Position = [141 34 70 22];

            % Create TimeSpacingsEditFieldLabel
            app.TimeSpacingsEditFieldLabel = uilabel(app.GeneralInformationPanel);
            app.TimeSpacingsEditFieldLabel.HorizontalAlignment = 'right';
            app.TimeSpacingsEditFieldLabel.Position = [2 7 95 22];
            app.TimeSpacingsEditFieldLabel.Text = 'Time Spacing (s)';

            % Create TimeSpacingsEditField
            app.TimeSpacingsEditField = uieditfield(app.GeneralInformationPanel, 'numeric');
            app.TimeSpacingsEditField.ValueDisplayFormat = '%5.2f';
            app.TimeSpacingsEditField.Position = [141 7 70 22];

            % Create GuideLinesPanel
            app.GuideLinesPanel = uipanel(app.TimedomainTab);
            app.GuideLinesPanel.AutoResizeChildren = 'off';
            app.GuideLinesPanel.Title = 'Guide Lines';
            app.GuideLinesPanel.Position = [15 381 216 62];

            % Create EnableButton
            app.EnableButton = uibutton(app.GuideLinesPanel, 'state');
            app.EnableButton.ValueChangedFcn = createCallbackFcn(app, @EnableButtonValueChanged, true);
            app.EnableButton.Text = 'Enable';
            app.EnableButton.Position = [10 9 92 23];

            % Create SetInitButton
            app.SetInitButton = uibutton(app.GuideLinesPanel, 'push');
            app.SetInitButton.ButtonPushedFcn = createCallbackFcn(app, @SetInitButtonPushed, true);
            app.SetInitButton.Enable = 'off';
            app.SetInitButton.Position = [113 9 92 23];
            app.SetInitButton.Text = 'Set Init';

            % Create ColormapPanel
            app.ColormapPanel = uipanel(app.TimedomainTab);
            app.ColormapPanel.AutoResizeChildren = 'off';
            app.ColormapPanel.Title = 'Colormap';
            app.ColormapPanel.Position = [15 451 216 85];

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

            % Create SpectrumPanel
            app.SpectrumPanel = uipanel(app.TimedomainTab);
            app.SpectrumPanel.AutoResizeChildren = 'off';
            app.SpectrumPanel.Title = 'Spectrum';
            app.SpectrumPanel.Position = [15 289 216 84];

            % Create SpectrogramButton
            app.SpectrogramButton = uibutton(app.SpectrumPanel, 'push');
            app.SpectrogramButton.ButtonPushedFcn = createCallbackFcn(app, @SpectrogramButtonPushed, true);
            app.SpectrogramButton.Enable = 'off';
            app.SpectrogramButton.Position = [10 10 92 23];
            app.SpectrogramButton.Text = 'Spectrogram';

            % Create PowerSpectrumButton
            app.PowerSpectrumButton = uibutton(app.SpectrumPanel, 'push');
            app.PowerSpectrumButton.ButtonPushedFcn = createCallbackFcn(app, @PowerSpectrumButtonPushed, true);
            app.PowerSpectrumButton.Enable = 'off';
            app.PowerSpectrumButton.Position = [112 10 100 23];
            app.PowerSpectrumButton.Text = 'Power Spectrum';

            % Create AutoScanSpectrumCheckBox
            app.AutoScanSpectrumCheckBox = uicheckbox(app.SpectrumPanel);
            app.AutoScanSpectrumCheckBox.Text = 'Auto-Scan';
            app.AutoScanSpectrumCheckBox.Position = [14 37 79 22];

            % Create AOIBoundaryTruncationPanel
            app.AOIBoundaryTruncationPanel = uipanel(app.TimedomainTab);
            app.AOIBoundaryTruncationPanel.AutoResizeChildren = 'off';
            app.AOIBoundaryTruncationPanel.Title = 'AOI Boundary Truncation';
            app.AOIBoundaryTruncationPanel.Position = [15 155 216 126];

            % Create TruncateButton
            app.TruncateButton = uibutton(app.AOIBoundaryTruncationPanel, 'push');
            app.TruncateButton.ButtonPushedFcn = createCallbackFcn(app, @TruncateButtonPushed, true);
            app.TruncateButton.BackgroundColor = [1 1 1];
            app.TruncateButton.FontWeight = 'bold';
            app.TruncateButton.FontColor = [0 0.4471 0.7412];
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

            % Create NextButton_2
            app.NextButton_2 = uibutton(app.TimedomainTab, 'push');
            app.NextButton_2.ButtonPushedFcn = createCallbackFcn(app, @NextButton_2Pushed, true);
            app.NextButton_2.Position = [22 15 92 25];
            app.NextButton_2.Text = 'Next';

            % Create SaveTruncatedButton
            app.SaveTruncatedButton = uibutton(app.TimedomainTab, 'push');
            app.SaveTruncatedButton.ButtonPushedFcn = createCallbackFcn(app, @SaveTruncatedButtonPushed, true);
            app.SaveTruncatedButton.Position = [127 15 89 25];
            app.SaveTruncatedButton.Text = 'THz Save';

            % Create ColormapcontrolPanel_TD
            app.ColormapcontrolPanel_TD = uipanel(app.TimedomainTab);
            app.ColormapcontrolPanel_TD.AutoResizeChildren = 'off';
            app.ColormapcontrolPanel_TD.Title = 'Colormap control';
            app.ColormapcontrolPanel_TD.Position = [15 544 216 80];

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
            app.SaveFigureButton = uibutton(app.TimedomainTab, 'push');
            app.SaveFigureButton.ButtonPushedFcn = createCallbackFcn(app, @SaveFigureButtonPushed, true);
            app.SaveFigureButton.Position = [758 16 95 23];
            app.SaveFigureButton.Text = 'Save Figure';

            % Create tXPickSlider
            app.tXPickSlider = uislider(app.TimedomainTab);
            app.tXPickSlider.ValueChangedFcn = createCallbackFcn(app, @tXPickSliderValueChanged, true);
            app.tXPickSlider.ValueChangingFcn = createCallbackFcn(app, @tXPickSliderValueChanging, true);
            app.tXPickSlider.Enable = 'off';
            app.tXPickSlider.Position = [921 250 383 3];

            % Create YlinesecLabel
            app.YlinesecLabel = uilabel(app.TimedomainTab);
            app.YlinesecLabel.HorizontalAlignment = 'right';
            app.YlinesecLabel.Enable = 'off';
            app.YlinesecLabel.Position = [1323 242 62 22];
            app.YlinesecLabel.Text = 'X line(sec)';

            % Create tXPickEditField
            app.tXPickEditField = uieditfield(app.TimedomainTab, 'numeric');
            app.tXPickEditField.Limits = [0 Inf];
            app.tXPickEditField.ValueDisplayFormat = '%5.2f';
            app.tXPickEditField.ValueChangedFcn = createCallbackFcn(app, @tXPickEditFieldValueChanged, true);
            app.tXPickEditField.Enable = 'off';
            app.tXPickEditField.Position = [1386 242 47 22];

            % Create tYPickSlider
            app.tYPickSlider = uislider(app.TimedomainTab);
            app.tYPickSlider.ValueChangedFcn = createCallbackFcn(app, @tYPickSliderValueChanged, true);
            app.tYPickSlider.ValueChangingFcn = createCallbackFcn(app, @tYPickSliderValueChanging, true);
            app.tYPickSlider.Enable = 'off';
            app.tYPickSlider.Position = [921 523 383 3];

            % Create tYPickEditField
            app.tYPickEditField = uieditfield(app.TimedomainTab, 'numeric');
            app.tYPickEditField.Limits = [0 Inf];
            app.tYPickEditField.ValueDisplayFormat = '%5.2f';
            app.tYPickEditField.ValueChangedFcn = createCallbackFcn(app, @tYPickEditFieldValueChanged, true);
            app.tYPickEditField.Enable = 'off';
            app.tYPickEditField.Position = [1385 519 47 22];

            % Create XlinesecLabel
            app.XlinesecLabel = uilabel(app.TimedomainTab);
            app.XlinesecLabel.HorizontalAlignment = 'right';
            app.XlinesecLabel.Enable = 'off';
            app.XlinesecLabel.Position = [1321 519 61 22];
            app.XlinesecLabel.Text = 'Y line(sec)';

            % Create SetLeftLimitButton
            app.SetLeftLimitButton = uibutton(app.TimedomainTab, 'push');
            app.SetLeftLimitButton.ButtonPushedFcn = createCallbackFcn(app, @SetLeftLimitButtonPushed, true);
            app.SetLeftLimitButton.Enable = 'off';
            app.SetLeftLimitButton.Position = [1337 214 100 23];
            app.SetLeftLimitButton.Text = 'Set Left Limit';

            % Create SetDownLimitButton
            app.SetDownLimitButton = uibutton(app.TimedomainTab, 'push');
            app.SetDownLimitButton.ButtonPushedFcn = createCallbackFcn(app, @SetDownLimitButtonPushed, true);
            app.SetDownLimitButton.Enable = 'off';
            app.SetDownLimitButton.Position = [1333 492 100 23];
            app.SetDownLimitButton.Text = 'Set Down Limit';

            % Create FrequencydomainTab
            app.FrequencydomainTab = uitab(app.TabGroup);
            app.FrequencydomainTab.AutoResizeChildren = 'off';
            app.FrequencydomainTab.Title = 'Frequency domain';

            % Create UIAxes21
            app.UIAxes21 = uiaxes(app.FrequencydomainTab);
            title(app.UIAxes21, 'Power spectrum')
            xlabel(app.UIAxes21, 'Time/Position (sec/mm)')
            ylabel(app.UIAxes21, 'Frequency (THz)')
            app.UIAxes21.PlotBoxAspectRatio = [1.0607476635514 1 1];
            app.UIAxes21.XTickLabelRotation = 0;
            app.UIAxes21.YTickLabelRotation = 0;
            app.UIAxes21.ZTickLabelRotation = 0;
            app.UIAxes21.Box = 'on';
            app.UIAxes21.FontSize = 11;
            app.UIAxes21.Position = [262 235 570 520];

            % Create UIAxes22
            app.UIAxes22 = uiaxes(app.FrequencydomainTab);
            title(app.UIAxes22, 'Phase')
            xlabel(app.UIAxes22, 'Time/Position (sec/mm)')
            ylabel(app.UIAxes22, 'Frequency (THz)')
            zlabel(app.UIAxes22, 'Time (ps)')
            app.UIAxes22.PlotBoxAspectRatio = [1.0607476635514 1 1];
            app.UIAxes22.ZDir = 'reverse';
            app.UIAxes22.XTickLabelRotation = 0;
            app.UIAxes22.YTickLabelRotation = 0;
            app.UIAxes22.ZTickLabelRotation = 0;
            app.UIAxes22.Box = 'on';
            app.UIAxes22.FontSize = 11;
            app.UIAxes22.Position = [840 235 570 520];

            % Create UIAxes23
            app.UIAxes23 = uiaxes(app.FrequencydomainTab);
            xlabel(app.UIAxes23, 'Frequency (THz)')
            ylabel(app.UIAxes23, 'E field (a.u.)')
            app.UIAxes23.PlotBoxAspectRatio = [3.546875 1 1];
            app.UIAxes23.Box = 'on';
            app.UIAxes23.FontSize = 11;
            app.UIAxes23.Position = [263 41 570 180];

            % Create UIAxes24
            app.UIAxes24 = uiaxes(app.FrequencydomainTab);
            xlabel(app.UIAxes24, 'Frequency (THz)')
            ylabel(app.UIAxes24, 'Phase')
            app.UIAxes24.PlotBoxAspectRatio = [3.546875 1 1];
            app.UIAxes24.Box = 'on';
            app.UIAxes24.FontSize = 11;
            app.UIAxes24.Position = [841 41 570 180];

            % Create FourierTransformPanel
            app.FourierTransformPanel = uipanel(app.FrequencydomainTab);
            app.FourierTransformPanel.AutoResizeChildren = 'off';
            app.FourierTransformPanel.Title = 'Fourier Transform';
            app.FourierTransformPanel.Position = [13 404 240 290];

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
            app.DataInformationPanel = uipanel(app.FrequencydomainTab);
            app.DataInformationPanel.AutoResizeChildren = 'off';
            app.DataInformationPanel.Title = 'Data Information';
            app.DataInformationPanel.Position = [13 700 239 55];

            % Create DataLengthEditField_2Label
            app.DataLengthEditField_2Label = uilabel(app.DataInformationPanel);
            app.DataLengthEditField_2Label.HorizontalAlignment = 'right';
            app.DataLengthEditField_2Label.Position = [7 8 70 22];
            app.DataLengthEditField_2Label.Text = ' DataLength';

            % Create dataLengthEditField_2
            app.dataLengthEditField_2 = uieditfield(app.DataInformationPanel, 'numeric');
            app.dataLengthEditField_2.ValueDisplayFormat = '%.0f';
            app.dataLengthEditField_2.Position = [86 9 40 20];

            % Create NumberofScansEditField_2Label
            app.NumberofScansEditField_2Label = uilabel(app.DataInformationPanel);
            app.NumberofScansEditField_2Label.HorizontalAlignment = 'right';
            app.NumberofScansEditField_2Label.Position = [132 8 48 22];
            app.NumberofScansEditField_2Label.Text = 'Number';

            % Create dataNumberEditField_2
            app.dataNumberEditField_2 = uieditfield(app.DataInformationPanel, 'numeric');
            app.dataNumberEditField_2.ValueDisplayFormat = '%.0f';
            app.dataNumberEditField_2.Position = [190 9 40 20];

            % Create NextButton
            app.NextButton = uibutton(app.FrequencydomainTab, 'push');
            app.NextButton.ButtonPushedFcn = createCallbackFcn(app, @NextButtonPushed, true);
            app.NextButton.FontWeight = 'bold';
            app.NextButton.Enable = 'off';
            app.NextButton.Position = [25 150 220 24];
            app.NextButton.Text = 'Next';

            % Create SingleMeasurementPanel
            app.SingleMeasurementPanel = uipanel(app.FrequencydomainTab);
            app.SingleMeasurementPanel.AutoResizeChildren = 'off';
            app.SingleMeasurementPanel.Title = 'Single Measurement';
            app.SingleMeasurementPanel.Position = [13 144 240 108];

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
            app.AssiginFFTDatainworkspaceButton = uibutton(app.FrequencydomainTab, 'push');
            app.AssiginFFTDatainworkspaceButton.ButtonPushedFcn = createCallbackFcn(app, @AssiginFFTDatainworkspaceButtonPushed, true);
            app.AssiginFFTDatainworkspaceButton.Position = [24 108 215 23];
            app.AssiginFFTDatainworkspaceButton.Text = 'Assigin FFT Data in workspace';

            % Create ColormapcontrolPanel
            app.ColormapcontrolPanel = uipanel(app.FrequencydomainTab);
            app.ColormapcontrolPanel.AutoResizeChildren = 'off';
            app.ColormapcontrolPanel.Title = 'Colormap control';
            app.ColormapcontrolPanel.Position = [13 257 239 141];

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

            % Create ExtractionTab
            app.ExtractionTab = uitab(app.TabGroup);
            app.ExtractionTab.AutoResizeChildren = 'off';
            app.ExtractionTab.Title = 'Extraction';

            % Create UIAxes31
            app.UIAxes31 = uiaxes(app.ExtractionTab);
            title(app.UIAxes31, 'Raw Measurement ')
            xlabel(app.UIAxes31, 'Time (sec)')
            ylabel(app.UIAxes31, 'Distance (mm)')
            app.UIAxes31.PlotBoxAspectRatio = [1.38414634146341 1 1];
            app.UIAxes31.Box = 'on';
            app.UIAxes31.FontSize = 11;
            app.UIAxes31.Position = [325 375 500 380];

            % Create UIAxes32
            app.UIAxes32 = uiaxes(app.ExtractionTab);
            title(app.UIAxes32, 'Filtered Measurement')
            xlabel(app.UIAxes32, 'Time (sec)')
            ylabel(app.UIAxes32, 'Distance (mm)')
            app.UIAxes32.PlotBoxAspectRatio = [1.38414634146341 1 1];
            app.UIAxes32.Box = 'on';
            app.UIAxes32.FontSize = 11;
            app.UIAxes32.Position = [825 375 500 380];

            % Create UIAxes33
            app.UIAxes33 = uiaxes(app.ExtractionTab);
            title(app.UIAxes33, 'Liquid Front Reflection')
            xlabel(app.UIAxes33, 'Time (sec)')
            ylabel(app.UIAxes33, 'E filed (a.u.)')
            app.UIAxes33.PlotBoxAspectRatio = [1.69402985074627 1 1];
            app.UIAxes33.Box = 'on';
            app.UIAxes33.FontSize = 11;
            app.UIAxes33.Position = [323 42 500 320];

            % Create UIAxes34
            app.UIAxes34 = uiaxes(app.ExtractionTab);
            title(app.UIAxes34, 'Liquid Front Ingress')
            xlabel(app.UIAxes34, 'Time (sec)')
            ylabel(app.UIAxes34, 'Distance (mm)')
            app.UIAxes34.PlotBoxAspectRatio = [1.69402985074627 1 1];
            app.UIAxes34.Box = 'on';
            app.UIAxes34.FontSize = 11;
            app.UIAxes34.Position = [825 42 500 320];

            % Create UIAxes35
            app.UIAxes35 = uiaxes(app.ExtractionTab);
            title(app.UIAxes35, 'Liquid Front ROI')
            xlabel(app.UIAxes35, 'Time (sec)')
            ylabel(app.UIAxes35, 'Distance (mm)')
            zlabel(app.UIAxes35, 'Z')
            app.UIAxes35.Box = 'on';
            app.UIAxes35.FontSize = 11;
            app.UIAxes35.Position = [16 53 292 219];

            % Create ExtractReflectionPointsPanel
            app.ExtractReflectionPointsPanel = uipanel(app.ExtractionTab);
            app.ExtractReflectionPointsPanel.AutoResizeChildren = 'off';
            app.ExtractReflectionPointsPanel.Title = 'Extract Reflection Points';
            app.ExtractReflectionPointsPanel.Position = [16 294 295 459];

            % Create AlphaDropDown_2Label
            app.AlphaDropDown_2Label = uilabel(app.ExtractReflectionPointsPanel);
            app.AlphaDropDown_2Label.HorizontalAlignment = 'right';
            app.AlphaDropDown_2Label.Position = [17 340 35 22];
            app.AlphaDropDown_2Label.Text = 'Alpha';

            % Create AlphaDropDown
            app.AlphaDropDown = uidropdown(app.ExtractReflectionPointsPanel);
            app.AlphaDropDown.Items = {'1.0', '0.7', '0.5', '0.3', '0.1'};
            app.AlphaDropDown.Position = [61 340 79 22];
            app.AlphaDropDown.Value = '1.0';

            % Create CmapLabel
            app.CmapLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.CmapLabel.HorizontalAlignment = 'right';
            app.CmapLabel.Position = [14 315 38 22];
            app.CmapLabel.Text = 'Cmap';

            % Create ExtColormapDropDown
            app.ExtColormapDropDown = uidropdown(app.ExtractReflectionPointsPanel);
            app.ExtColormapDropDown.Items = {'parula', 'jet', 'copper', 'bone', 'hot'};
            app.ExtColormapDropDown.Position = [61 315 80 22];
            app.ExtColormapDropDown.Value = 'parula';

            % Create LfPlotButton
            app.LfPlotButton = uibutton(app.ExtractReflectionPointsPanel, 'push');
            app.LfPlotButton.ButtonPushedFcn = createCallbackFcn(app, @LfPlotButtonPushed, true);
            app.LfPlotButton.Position = [151 316 122 47];
            app.LfPlotButton.Text = 'Replot';

            % Create ExtractLiquidFrontButton
            app.ExtractLiquidFrontButton = uibutton(app.ExtractReflectionPointsPanel, 'push');
            app.ExtractLiquidFrontButton.ButtonPushedFcn = createCallbackFcn(app, @ExtractLiquidFrontButtonPushed, true);
            app.ExtractLiquidFrontButton.FontWeight = 'bold';
            app.ExtractLiquidFrontButton.Position = [20 91 258 30];
            app.ExtractLiquidFrontButton.Text = 'Extract Liquid Front';

            % Create SampleNameEditField_T3Label
            app.SampleNameEditField_T3Label = uilabel(app.ExtractReflectionPointsPanel);
            app.SampleNameEditField_T3Label.HorizontalAlignment = 'right';
            app.SampleNameEditField_T3Label.Position = [7 414 110 22];
            app.SampleNameEditField_T3Label.Text = 'Sample Description';

            % Create SampleDescriptionEditField_T3
            app.SampleDescriptionEditField_T3 = uieditfield(app.ExtractReflectionPointsPanel, 'text');
            app.SampleDescriptionEditField_T3.Position = [11 394 268 22];

            % Create CaculateIngressTimeButton
            app.CaculateIngressTimeButton = uibutton(app.ExtractReflectionPointsPanel, 'push');
            app.CaculateIngressTimeButton.ButtonPushedFcn = createCallbackFcn(app, @CaculateIngressTimeButtonPushed, true);
            app.CaculateIngressTimeButton.Position = [19 33 259 30];
            app.CaculateIngressTimeButton.Text = 'Caculate Ingress Time';

            % Create n_effLabel
            app.n_effLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.n_effLabel.HorizontalAlignment = 'right';
            app.n_effLabel.Position = [195 368 32 22];
            app.n_effLabel.Text = 'n_eff';

            % Create RefractiveIndexEditField_T3
            app.RefractiveIndexEditField_T3 = uieditfield(app.ExtractReflectionPointsPanel, 'numeric');
            app.RefractiveIndexEditField_T3.Limits = [0 Inf];
            app.RefractiveIndexEditField_T3.ValueDisplayFormat = '%5.2f';
            app.RefractiveIndexEditField_T3.Editable = 'off';
            app.RefractiveIndexEditField_T3.Position = [236 368 40 22];

            % Create ALGORITHMROIButton
            app.ALGORITHMROIButton = uibutton(app.ExtractReflectionPointsPanel, 'push');
            app.ALGORITHMROIButton.ButtonPushedFcn = createCallbackFcn(app, @ALGORITHMROIButtonPushed, true);
            app.ALGORITHMROIButton.Position = [21 160 125 30];
            app.ALGORITHMROIButton.Text = 'ALGORITHM ROI';

            % Create DistancetoCentrepsEditFieldLabel
            app.DistancetoCentrepsEditFieldLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.DistancetoCentrepsEditFieldLabel.HorizontalAlignment = 'right';
            app.DistancetoCentrepsEditFieldLabel.Position = [18 368 128 22];
            app.DistancetoCentrepsEditFieldLabel.Text = 'Distance to Centre (ps)';

            % Create DistancetoCentrepsEditField
            app.DistancetoCentrepsEditField = uieditfield(app.ExtractReflectionPointsPanel, 'numeric');
            app.DistancetoCentrepsEditField.Limits = [0 Inf];
            app.DistancetoCentrepsEditField.ValueDisplayFormat = '%5.2f';
            app.DistancetoCentrepsEditField.Position = [150 368 40 22];

            % Create DBSCANNeighborhoodRadiusSwitchLabel
            app.DBSCANNeighborhoodRadiusSwitchLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.DBSCANNeighborhoodRadiusSwitchLabel.HorizontalAlignment = 'center';
            app.DBSCANNeighborhoodRadiusSwitchLabel.Position = [20 220 175 22];
            app.DBSCANNeighborhoodRadiusSwitchLabel.Text = 'DBSCAN Neighborhood Radius';

            % Create DBSCANNeighborhoodRadiusSwitch
            app.DBSCANNeighborhoodRadiusSwitch = uiswitch(app.ExtractReflectionPointsPanel, 'slider');
            app.DBSCANNeighborhoodRadiusSwitch.Items = {'Auto', 'Manual'};
            app.DBSCANNeighborhoodRadiusSwitch.ValueChangedFcn = createCallbackFcn(app, @DBSCANNeighborhoodRadiusSwitchValueChanged, true);
            app.DBSCANNeighborhoodRadiusSwitch.Position = [59 199 45 20];
            app.DBSCANNeighborhoodRadiusSwitch.Value = 'Auto';

            % Create Range220EditFieldLabel
            app.Range220EditFieldLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.Range220EditFieldLabel.HorizontalAlignment = 'right';
            app.Range220EditFieldLabel.Position = [153 199 76 22];
            app.Range220EditFieldLabel.Text = '(Range 2-20)';

            % Create Range220EditField
            app.Range220EditField = uieditfield(app.ExtractReflectionPointsPanel, 'numeric');
            app.Range220EditField.Limits = [2 20];
            app.Range220EditField.ValueDisplayFormat = '%.0f';
            app.Range220EditField.Editable = 'off';
            app.Range220EditField.Position = [235 199 30 22];
            app.Range220EditField.Value = 10;

            % Create LiquidIngressTimesecEditFieldLabel
            app.LiquidIngressTimesecEditFieldLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.LiquidIngressTimesecEditFieldLabel.HorizontalAlignment = 'right';
            app.LiquidIngressTimesecEditFieldLabel.Position = [62 6 139 22];
            app.LiquidIngressTimesecEditFieldLabel.Text = 'Liquid Ingress Time (sec)';

            % Create LiquidIngressTimesecEditField
            app.LiquidIngressTimesecEditField = uieditfield(app.ExtractReflectionPointsPanel, 'numeric');
            app.LiquidIngressTimesecEditField.Limits = [0 Inf];
            app.LiquidIngressTimesecEditField.ValueDisplayFormat = '%5.2f';
            app.LiquidIngressTimesecEditField.Editable = 'off';
            app.LiquidIngressTimesecEditField.Position = [208 6 64 22];

            % Create ExcludeLowerReflectionsCheckBox
            app.ExcludeLowerReflectionsCheckBox = uicheckbox(app.ExtractReflectionPointsPanel);
            app.ExcludeLowerReflectionsCheckBox.Text = 'Exclude Lower Reflections';
            app.ExcludeLowerReflectionsCheckBox.Position = [29 65 163 22];

            % Create ROITheasholdSliderLabel
            app.ROITheasholdSliderLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.ROITheasholdSliderLabel.HorizontalAlignment = 'right';
            app.ROITheasholdSliderLabel.Position = [13 286 86 22];
            app.ROITheasholdSliderLabel.Text = 'ROI Theashold';

            % Create ROITheasholdSlider
            app.ROITheasholdSlider = uislider(app.ExtractReflectionPointsPanel);
            app.ROITheasholdSlider.ValueChangedFcn = createCallbackFcn(app, @ROITheasholdSliderValueChanged, true);
            app.ROITheasholdSlider.Position = [44 275 210 3];

            % Create ExcludeuptopsEditFieldLabel
            app.ExcludeuptopsEditFieldLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.ExcludeuptopsEditFieldLabel.HorizontalAlignment = 'right';
            app.ExcludeuptopsEditFieldLabel.Position = [133 285 98 22];
            app.ExcludeuptopsEditFieldLabel.Text = 'Exclude upto (ps)';

            % Create ExcludeuptopsEditField
            app.ExcludeuptopsEditField = uieditfield(app.ExtractReflectionPointsPanel, 'numeric');
            app.ExcludeuptopsEditField.Limits = [0 3];
            app.ExcludeuptopsEditField.ValueDisplayFormat = '%3.1f';
            app.ExcludeuptopsEditField.ValueChangedFcn = createCallbackFcn(app, @ExcludeuptopsEditFieldValueChanged, true);
            app.ExcludeuptopsEditField.Position = [235 285 30 22];

            % Create FREEHANDROIButton
            app.FREEHANDROIButton = uibutton(app.ExtractReflectionPointsPanel, 'push');
            app.FREEHANDROIButton.ButtonPushedFcn = createCallbackFcn(app, @FREEHANDROIButtonPushed, true);
            app.FREEHANDROIButton.Position = [153 160 125 30];
            app.FREEHANDROIButton.Text = 'FREEHAND ROI';

            % Create ROIDropDownLabel
            app.ROIDropDownLabel = uilabel(app.ExtractReflectionPointsPanel);
            app.ROIDropDownLabel.HorizontalAlignment = 'right';
            app.ROIDropDownLabel.Position = [23 130 27 22];
            app.ROIDropDownLabel.Text = 'ROI';

            % Create ROIDropDown
            app.ROIDropDown = uidropdown(app.ExtractReflectionPointsPanel);
            app.ROIDropDown.Items = {'ALGORITHM ONLY', 'FREEHAND ONLY', 'ALGORITHM + FREEHAND'};
            app.ROIDropDown.ValueChangedFcn = createCallbackFcn(app, @ROIDropDownValueChanged, true);
            app.ROIDropDown.Position = [59 130 220 22];
            app.ROIDropDown.Value = 'ALGORITHM ONLY';

            % Create BatchManagementButton
            app.BatchManagementButton = uibutton(app.ExtractionTab, 'push');
            app.BatchManagementButton.ButtonPushedFcn = createCallbackFcn(app, @BatchManagementButtonPushed, true);
            app.BatchManagementButton.Position = [32 3 246 30];
            app.BatchManagementButton.Text = 'Batch Management';

            % Create BatchAnalysisTab
            app.BatchAnalysisTab = uitab(app.TabGroup);
            app.BatchAnalysisTab.AutoResizeChildren = 'off';
            app.BatchAnalysisTab.Title = 'Batch Analysis';

            % Create UIAxes42
            app.UIAxes42 = uiaxes(app.BatchAnalysisTab);
            title(app.UIAxes42, 'Liquid Front Ingress')
            xlabel(app.UIAxes42, 'Time (sec)')
            ylabel(app.UIAxes42, 'Displacement (mm)')
            app.UIAxes42.Box = 'on';
            app.UIAxes42.FontSize = 11;
            app.UIAxes42.Position = [863 39 470 400];

            % Create UIAxes41
            app.UIAxes41 = uiaxes(app.BatchAnalysisTab);
            title(app.UIAxes41, 'Liquid Front Reflection')
            xlabel(app.UIAxes41, 'Time (sec)')
            ylabel(app.UIAxes41, 'E field intensity (a.u.)')
            app.UIAxes41.Box = 'on';
            app.UIAxes41.FontSize = 11;
            app.UIAxes41.Position = [383 39 470 400];

            % Create GroupButton
            app.GroupButton = uibutton(app.BatchAnalysisTab, 'push');
            app.GroupButton.ButtonPushedFcn = createCallbackFcn(app, @GroupButtonPushed, true);
            app.GroupButton.Position = [244 657 128 30];
            app.GroupButton.Text = 'Group';

            % Create UngroupButton
            app.UngroupButton = uibutton(app.BatchAnalysisTab, 'push');
            app.UngroupButton.ButtonPushedFcn = createCallbackFcn(app, @UngroupButtonPushed, true);
            app.UngroupButton.Position = [243 616 129 30];
            app.UngroupButton.Text = 'Ungroup';

            % Create RemoveButton
            app.RemoveButton = uibutton(app.BatchAnalysisTab, 'push');
            app.RemoveButton.ButtonPushedFcn = createCallbackFcn(app, @RemoveButtonPushed, true);
            app.RemoveButton.Position = [557 461 126 29];
            app.RemoveButton.Text = 'Remove';

            % Create InformationPanel
            app.InformationPanel = uipanel(app.BatchAnalysisTab);
            app.InformationPanel.AutoResizeChildren = 'off';
            app.InformationPanel.Title = 'Information';
            app.InformationPanel.Position = [15 10 227 223];

            % Create DistancetoCentremmLabel
            app.DistancetoCentremmLabel = uilabel(app.InformationPanel);
            app.DistancetoCentremmLabel.HorizontalAlignment = 'right';
            app.DistancetoCentremmLabel.Position = [16 95 128 22];
            app.DistancetoCentremmLabel.Text = 'Distance to Centre (ps)';

            % Create BIDistanceToCentreEditField
            app.BIDistanceToCentreEditField = uieditfield(app.InformationPanel, 'numeric');
            app.BIDistanceToCentreEditField.Position = [167 96 52 22];

            % Create SampleDescriptionLabel
            app.SampleDescriptionLabel = uilabel(app.InformationPanel);
            app.SampleDescriptionLabel.HorizontalAlignment = 'right';
            app.SampleDescriptionLabel.Position = [10 176 110 22];
            app.SampleDescriptionLabel.Text = 'Sample Description';

            % Create BIDescriptionEditField
            app.BIDescriptionEditField = uieditfield(app.InformationPanel, 'text');
            app.BIDescriptionEditField.ValueChangedFcn = createCallbackFcn(app, @BIDescriptionEditFieldValueChanged, true);
            app.BIDescriptionEditField.Position = [7 154 211 22];

            % Create RefractiveIndexEditField_2Label
            app.RefractiveIndexEditField_2Label = uilabel(app.InformationPanel);
            app.RefractiveIndexEditField_2Label.HorizontalAlignment = 'right';
            app.RefractiveIndexEditField_2Label.Position = [8 67 92 22];
            app.RefractiveIndexEditField_2Label.Text = 'Refractive Index';

            % Create BIRefractiveIndexEditField
            app.BIRefractiveIndexEditField = uieditfield(app.InformationPanel, 'numeric');
            app.BIRefractiveIndexEditField.Position = [167 67 52 22];

            % Create IngressTimesecEditFieldLabel
            app.IngressTimesecEditFieldLabel = uilabel(app.InformationPanel);
            app.IngressTimesecEditFieldLabel.HorizontalAlignment = 'right';
            app.IngressTimesecEditFieldLabel.Position = [8 39 104 22];
            app.IngressTimesecEditFieldLabel.Text = 'Ingress Time (sec)';

            % Create BIIngressTimesecEditField
            app.BIIngressTimesecEditField = uieditfield(app.InformationPanel, 'numeric');
            app.BIIngressTimesecEditField.Position = [167 39 52 22];

            % Create MeanSpeedmmminLabel
            app.MeanSpeedmmminLabel = uilabel(app.InformationPanel);
            app.MeanSpeedmmminLabel.HorizontalAlignment = 'right';
            app.MeanSpeedmmminLabel.Position = [8 12 127 22];
            app.MeanSpeedmmminLabel.Text = 'Mean Speed (mm/min)';

            % Create BIMeanSpeedEditField
            app.BIMeanSpeedEditField = uieditfield(app.InformationPanel, 'numeric');
            app.BIMeanSpeedEditField.Position = [167 12 52 22];

            % Create BatchLabel
            app.BatchLabel = uilabel(app.InformationPanel);
            app.BatchLabel.HorizontalAlignment = 'right';
            app.BatchLabel.Position = [10 124 36 22];
            app.BatchLabel.Text = 'Batch';

            % Create BIBatchEditField
            app.BIBatchEditField = uieditfield(app.InformationPanel, 'text');
            app.BIBatchEditField.Position = [55 124 163 22];

            % Create MeasurementListBoxLabel
            app.MeasurementListBoxLabel = uilabel(app.BatchAnalysisTab);
            app.MeasurementListBoxLabel.HorizontalAlignment = 'right';
            app.MeasurementListBoxLabel.Position = [18 722 79 22];
            app.MeasurementListBoxLabel.Text = 'Measurement';

            % Create MeasurementListBox
            app.MeasurementListBox = uilistbox(app.BatchAnalysisTab);
            app.MeasurementListBox.Items = {};
            app.MeasurementListBox.Multiselect = 'on';
            app.MeasurementListBox.ValueChangedFcn = createCallbackFcn(app, @MeasurementListBoxValueChanged, true);
            app.MeasurementListBox.Position = [18 357 206 364];
            app.MeasurementListBox.Value = {};

            % Create BatchNameEditFieldLabel
            app.BatchNameEditFieldLabel = uilabel(app.BatchAnalysisTab);
            app.BatchNameEditFieldLabel.HorizontalAlignment = 'right';
            app.BatchNameEditFieldLabel.Position = [247 720 72 22];
            app.BatchNameEditFieldLabel.Text = 'Batch Name';

            % Create BatchNameEditField
            app.BatchNameEditField = uieditfield(app.BatchAnalysisTab, 'text');
            app.BatchNameEditField.Position = [245 695 127 25];

            % Create BatchListBoxLabel
            app.BatchListBoxLabel = uilabel(app.BatchAnalysisTab);
            app.BatchListBoxLabel.HorizontalAlignment = 'right';
            app.BatchListBoxLabel.Position = [391 722 33 22];
            app.BatchListBoxLabel.Text = 'Batch';

            % Create BatchListBox
            app.BatchListBox = uilistbox(app.BatchAnalysisTab);
            app.BatchListBox.Items = {};
            app.BatchListBox.Multiselect = 'on';
            app.BatchListBox.ValueChangedFcn = createCallbackFcn(app, @BatchListBoxValueChanged, true);
            app.BatchListBox.Position = [389 461 148 260];
            app.BatchListBox.Value = {};

            % Create BatchDetailListBoxLabel
            app.BatchDetailListBoxLabel = uilabel(app.BatchAnalysisTab);
            app.BatchDetailListBoxLabel.HorizontalAlignment = 'right';
            app.BatchDetailListBoxLabel.Position = [552 723 70 22];
            app.BatchDetailListBoxLabel.Text = 'Batch Detail';

            % Create BatchDetailListBox
            app.BatchDetailListBox = uilistbox(app.BatchAnalysisTab);
            app.BatchDetailListBox.Items = {};
            app.BatchDetailListBox.Position = [549 498 142 223];
            app.BatchDetailListBox.Value = {};

            % Create RemoveButton_2
            app.RemoveButton_2 = uibutton(app.BatchAnalysisTab, 'push');
            app.RemoveButton_2.ButtonPushedFcn = createCallbackFcn(app, @RemoveButton_2Pushed, true);
            app.RemoveButton_2.Position = [28 319 182 29];
            app.RemoveButton_2.Text = 'Remove';

            % Create AddButton
            app.AddButton = uibutton(app.BatchAnalysisTab, 'push');
            app.AddButton.ButtonPushedFcn = createCallbackFcn(app, @AddButtonPushed, true);
            app.AddButton.Position = [243 576 129 30];
            app.AddButton.Text = 'Add';

            % Create AssigndatainworkspaceButton
            app.AssigndatainworkspaceButton = uibutton(app.BatchAnalysisTab, 'push');
            app.AssigndatainworkspaceButton.ButtonPushedFcn = createCallbackFcn(app, @AssigndatainworkspaceButtonPushed, true);
            app.AssigndatainworkspaceButton.Position = [22 43 158 32];
            app.AssigndatainworkspaceButton.Text = 'Assign data in workspace';

            % Create SaveProjectButton
            app.SaveProjectButton = uibutton(app.BatchAnalysisTab, 'push');
            app.SaveProjectButton.ButtonPushedFcn = createCallbackFcn(app, @SaveProjectButtonPushed, true);
            app.SaveProjectButton.Position = [189 43 78 32];
            app.SaveProjectButton.Text = 'Save Project';

            % Create LoadProjectButton
            app.LoadProjectButton = uibutton(app.BatchAnalysisTab, 'push');
            app.LoadProjectButton.ButtonPushedFcn = createCallbackFcn(app, @LoadProjectButtonPushed, true);
            app.LoadProjectButton.Position = [273 43 78 32];
            app.LoadProjectButton.Text = 'Load Project';

            % Create UITable
            app.UITable = uitable(app.BatchAnalysisTab);
            app.UITable.ColumnName = {'Description'; 'Ingress Time'; 'Y-intercept'; 'para01'; 'para02'; 'd'; 'R^2'; 'RMSE'};
            app.UITable.ColumnWidth = {'auto', 60, 60, 60, 60, 60, 60, 60};
            app.UITable.RowName = {};
            app.UITable.Position = [706 463 600 257];

            % Create FittingFunctionParametersunderdevelopmentLabel
            app.FittingFunctionParametersunderdevelopmentLabel = uilabel(app.BatchAnalysisTab);
            app.FittingFunctionParametersunderdevelopmentLabel.Position = [710 723 270 22];
            app.FittingFunctionParametersunderdevelopmentLabel.Text = 'Fitting Function Parameters (under development)';

            % Create DisplayButtonGroup
            app.DisplayButtonGroup = uibuttongroup(app.BatchAnalysisTab);
            app.DisplayButtonGroup.AutoResizeChildren = 'off';
            app.DisplayButtonGroup.Title = 'Display';
            app.DisplayButtonGroup.Position = [251 282 120 72];

            % Create IndividualButton
            app.IndividualButton = uiradiobutton(app.DisplayButtonGroup);
            app.IndividualButton.Text = 'Individual';
            app.IndividualButton.Position = [11 26 73 22];
            app.IndividualButton.Value = true;

            % Create BatchButton
            app.BatchButton = uiradiobutton(app.DisplayButtonGroup);
            app.BatchButton.Text = 'Batch';
            app.BatchButton.Position = [11 4 54 22];

            % Create StyleButtonGroup
            app.StyleButtonGroup = uibuttongroup(app.BatchAnalysisTab);
            app.StyleButtonGroup.AutoResizeChildren = 'off';
            app.StyleButtonGroup.Title = 'Style';
            app.StyleButtonGroup.Position = [251 175 120 100];

            % Create AllRangeButton
            app.AllRangeButton = uiradiobutton(app.StyleButtonGroup);
            app.AllRangeButton.Text = 'All Range';
            app.AllRangeButton.Position = [11 54 73 22];
            app.AllRangeButton.Value = true;

            % Create SDShadowedButton
            app.SDShadowedButton = uiradiobutton(app.StyleButtonGroup);
            app.SDShadowedButton.Text = 'SD (Shadowed)';
            app.SDShadowedButton.Position = [11 32 105 22];

            % Create SDErrorBarButton
            app.SDErrorBarButton = uiradiobutton(app.StyleButtonGroup);
            app.SDErrorBarButton.Text = 'SD (ErrorBar)';
            app.SDErrorBarButton.Position = [11 8 92 22];

            % Create LegendCheckBox
            app.LegendCheckBox = uicheckbox(app.BatchAnalysisTab);
            app.LegendCheckBox.Text = 'Legend';
            app.LegendCheckBox.Position = [261 226 62 22];
            app.LegendCheckBox.Value = true;

            % Create FittingCheckBox
            app.FittingCheckBox = uicheckbox(app.BatchAnalysisTab);
            app.FittingCheckBox.Text = ' Fitting';
            app.FittingCheckBox.Position = [261 202 63 22];

            % Create PlotButton_2
            app.PlotButton_2 = uibutton(app.BatchAnalysisTab, 'push');
            app.PlotButton_2.ButtonPushedFcn = createCallbackFcn(app, @PlotButton_2Pushed, true);
            app.PlotButton_2.Position = [256 154 109 42];
            app.PlotButton_2.Text = 'Plot';

            % Create DPlotNewButton
            app.DPlotNewButton = uibutton(app.BatchAnalysisTab, 'push');
            app.DPlotNewButton.ButtonPushedFcn = createCallbackFcn(app, @DPlotNewButtonPushed, true);
            app.DPlotNewButton.Position = [254 120 114 27];
            app.DPlotNewButton.Text = '3D Plot (New)';

            % Create ColourPlotNewButton
            app.ColourPlotNewButton = uibutton(app.BatchAnalysisTab, 'push');
            app.ColourPlotNewButton.ButtonPushedFcn = createCallbackFcn(app, @ColourPlotNewButtonPushed, true);
            app.ColourPlotNewButton.Position = [254 87 114 27];
            app.ColourPlotNewButton.Text = 'Colour Plot (New)';

            % Create DatasettoUseButtonGroup
            app.DatasettoUseButtonGroup = uibuttongroup(app.BatchAnalysisTab);
            app.DatasettoUseButtonGroup.AutoResizeChildren = 'off';
            app.DatasettoUseButtonGroup.Title = 'Dataset to Use';
            app.DatasettoUseButtonGroup.Position = [251 361 120 74];

            % Create ScatteredButton
            app.ScatteredButton = uiradiobutton(app.DatasettoUseButtonGroup);
            app.ScatteredButton.Text = 'Scattered';
            app.ScatteredButton.Position = [11 28 73 22];
            app.ScatteredButton.Value = true;

            % Create MeanButton
            app.MeanButton = uiradiobutton(app.DatasettoUseButtonGroup);
            app.MeanButton.Text = 'Mean';
            app.MeanButton.Position = [11 6 65 22];

            % Create StopProcessButton
            app.StopProcessButton = uibutton(app.DipTabInsightUIFigure, 'push');
            app.StopProcessButton.ButtonPushedFcn = createCallbackFcn(app, @StopProcessButtonPushed, true);
            app.StopProcessButton.FontWeight = 'bold';
            app.StopProcessButton.Position = [959 11 110 23];
            app.StopProcessButton.Text = 'Stop Process';

            % Create Image
            app.Image = uiimage(app.DipTabInsightUIFigure);
            app.Image.Position = [19 846 67 60];
            app.Image.ImageSource = fullfile(pathToMLAPP, 'Images', 'dotTHz_logo.png');

            % Create DipTabInsightLabel
            app.DipTabInsightLabel = uilabel(app.DipTabInsightUIFigure);
            app.DipTabInsightLabel.FontSize = 32;
            app.DipTabInsightLabel.FontWeight = 'bold';
            app.DipTabInsightLabel.FontAngle = 'italic';
            app.DipTabInsightLabel.FontColor = [0.149 0.149 0.149];
            app.DipTabInsightLabel.Position = [85 859 228 47];
            app.DipTabInsightLabel.Text = 'DipTab Insight';

            % Create DeployButton
            app.DeployButton = uibutton(app.DipTabInsightUIFigure, 'push');
            app.DeployButton.ButtonPushedFcn = createCallbackFcn(app, @DeployButtonPushed, true);
            app.DeployButton.FontWeight = 'bold';
            app.DeployButton.Position = [907 850 114 28];
            app.DeployButton.Text = 'Deploy';

            % Create DescriptionEditFieldLabel
            app.DescriptionEditFieldLabel = uilabel(app.DipTabInsightUIFigure);
            app.DescriptionEditFieldLabel.HorizontalAlignment = 'right';
            app.DescriptionEditFieldLabel.Position = [1043 852 66 22];
            app.DescriptionEditFieldLabel.Text = 'Description';

            % Create SampleDescriptionEditField
            app.SampleDescriptionEditField = uieditfield(app.DipTabInsightUIFigure, 'text');
            app.SampleDescriptionEditField.HorizontalAlignment = 'right';
            app.SampleDescriptionEditField.Position = [1119 853 303 20];

            % Show the figure after all components are created
            app.DipTabInsightUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = DipTabInsight_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.DipTabInsightUIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.DipTabInsightUIFigure)
        end
    end
end