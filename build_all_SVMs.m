function build_all_SVMs()
sides_to_process_all = {
    'both_series'
    %     'both_series'
    %     'both_series'
    };
normalize_flag = [
    1
    %     1
    %     0
    ];
ortho_flag = [
    0
    %     1
    %     0
    ];

params_table = table(sides_to_process_all,normalize_flag,ortho_flag,'VariableNames',...
    {'sides_to_process_all','normalize_flag','ortho_flag'});

power_band_filenames = {
%     'workspace_power_band_012'
    'workspace_power_band_011'
%     'workspace_power_band_010'
%     'workspace_power_band_009'
%     'workspace_power_band_008'
%     'workspace_power_band_007'
%     'workspace_power_band_006'
%     'workspace_power_band_005'
    };

for filename_idx = 1:length(power_band_filenames)
    band_data_base_filename = power_band_filenames{filename_idx}; 
    band_data_filename = ['../M004/',band_data_base_filename,'.mat'];

    for model_idx = 1:size(params_table,1)
        
        side_to_process = params_table.sides_to_process_all{model_idx};
        
        %% Seed random number generation for reproducibility
        rng('default')
        
        %% extract relevant subset of band data
        load(band_data_filename,...
            'band_data_table','frequency_bands_to_use','band_names')
        
        unique_subject_ID = unique(band_data_table.original_numbering);
        training_and_testing_data = cell(length(unique_subject_ID),2);
        for subject_idx = 1:length(unique_subject_ID)
            band_data_off_left = band_data_table.band_power{find(contains(band_data_table.side,'left') & ...
                contains(band_data_table.med_condition,'off') & ...
                band_data_table.original_numbering==unique_subject_ID(subject_idx))};
            
            band_data_on_left = band_data_table.band_power{find(contains(band_data_table.side,'left') & ...
                contains(band_data_table.med_condition,'on') & ...
                band_data_table.original_numbering==unique_subject_ID(subject_idx))};
            
            band_data_off_right = band_data_table.band_power{find(contains(band_data_table.side,'right') & ...
                contains(band_data_table.med_condition,'off') & ...
                band_data_table.original_numbering==unique_subject_ID(subject_idx))};
            
            band_data_on_right = band_data_table.band_power{find(contains(band_data_table.side,'right') & ...
                contains(band_data_table.med_condition,'on') & ...
                band_data_table.original_numbering==unique_subject_ID(subject_idx))};
            
            
            %%% balance on and off datasets by trimming out some of the data to even the two datasets out
            size_of_smaller_dataset = min(size(band_data_off_left,2),size(band_data_on_left,2));
            n_features = size(band_data_off_left,1);
            assert(size(band_data_off_left,1)==size(band_data_on_left,1))
            
            band_data_off_left_size_balanced = band_data_off_left(:,1:size_of_smaller_dataset); 
            band_data_off_right_size_balanced = band_data_off_right(:,1:size_of_smaller_dataset); 
            
            band_data_on_left_size_balanced = band_data_on_left(:,1:size_of_smaller_dataset); 
            band_data_on_right_size_balanced = band_data_on_right(:,1:size_of_smaller_dataset); 
            
            if (strcmp(side_to_process,'left')) % just left hemisphere data
                off_power = band_data_off_left_size_balanced;
                on_power = band_data_on_left_size_balanced;
                workspace_output_name = 'workspace_SVMs_left_hemispheres.mat';
            elseif (strcmp(side_to_process,'right')) % just right hemisphere data
                off_power = band_data_off_right_size_balanced;
                on_power = band_data_on_right_size_balanced;
                workspace_output_name = 'workspace_SVMs_right_hemispheres.mat';
            elseif (strcmp(side_to_process,'both_series')) % treates left and right hemispheres as equivalent (so preserves number of features)
%                 off_power = [band_data_off_left_size_balanced,band_data_off_right_size_balanced];
%                 on_power = [band_data_on_left_size_balanced,band_data_on_right_size_balanced];
                % stitch the data across hemispheres such that the
                % adjacent data points are also adjacent in time
                off_power = reshape([band_data_off_left_size_balanced;band_data_off_right_size_balanced],n_features,[]);
                on_power = reshape([band_data_on_left_size_balanced;band_data_on_right_size_balanced],n_features,[]);
                workspace_output_name = 'workspace_SVMs_series_hemispheres_combo.mat';
            elseif (strcmp(side_to_process,'both_parallel')) % treates each hemisphere as a separate feature (so doubles number of features)
                off_power = [band_data_off_left_size_balanced;band_data_off_right_size_balanced];
                on_power = [band_data_on_left_size_balanced;band_data_on_right_size_balanced];
                workspace_output_name = 'workspace_SVMs_parallel_hemispheres_combo.mat';
            else
                error('Invalid value for side_to_process...')
            end
            
            %%% balance on and off datasets by trimming out some of the data to even the two datasets out
            assert(size(off_power,2)==size(on_power,2),'sizes should already be balanced in preceding lines...')
            size_of_smaller_dataset = min(size(off_power,2),size(on_power,2));
            off_power_size_balanced = off_power(:,1:size_of_smaller_dataset); %randperm(size(off_power,2),size_of_smaller_dataset));
            on_power_size_balanced = on_power(:,1:size_of_smaller_dataset); %randperm(size(on_power,2),size_of_smaller_dataset));
            
            X = [off_power_size_balanced, on_power_size_balanced]';
            Y = [repmat({'off'},size(off_power_size_balanced,2),1); ...
                repmat({'on'},size(on_power_size_balanced,2),1) ];
            
            flag_normalize = params_table.normalize_flag(model_idx);
            flag_orthogonalize = params_table.ortho_flag(model_idx);
            if (flag_normalize)
                normalized_X = normalize_signal(X);
                X = normalized_X;
                workspace_output_name = strrep(workspace_output_name,'.mat','_norm.mat');
            end
            if (flag_orthogonalize)
                [R,P] = corrcoef(X);
                [U,D,V] = svd(R);
                orthognalized_X = ((U')*(X'))';
                norm_ortho_X = normalize_signal(orthognalized_X);
                X = norm_ortho_X;
                workspace_output_name = strrep(workspace_output_name,'.mat','_ortho.mat');
            end
            
            training_and_testing_data{subject_idx,1} = X;
            training_and_testing_data{subject_idx,2} = Y;
            
            
            XY_data_dir = sprintf('%s/%s_XY_data',band_data_base_filename,strrep(workspace_output_name,'.mat',''));
            if (~exist(XY_data_dir,'dir'))
                mkdir(XY_data_dir)
            end
            save(sprintf('%s/PD_%02d.mat',XY_data_dir,unique_subject_ID(subject_idx)),'X','Y')
        end
    end
end

%% Calculate the best hyperplane that separates the data points
for filename_idx = 1:length(power_band_filenames)
    band_data_base_filename = power_band_filenames{filename_idx}; 

    fprintf('Processing %s...\n',band_data_base_filename);
    for model_idx = 1:size(params_table,1)
        
        K_FOLD = 10;
        
        svm_models = cell(length(unique_subject_ID),1);
        optimal_box_constraint_all_subjects = zeros(length(unique_subject_ID),1);
        bayesopt_results_all = cell(length(unique_subject_ID),1);
        
        %         h = waitbar(0);
        
        SVM_dir = sprintf('%s/%s_SVMs',band_data_base_filename,strrep(workspace_output_name,'.mat',''));
        if (~exist(SVM_dir,'dir'))
            mkdir(SVM_dir)
        end
        
        c = parcluster;
        if (ispc())
            n_workers = 2;
        else
            n_workers = min(c.NumWorkers,17);
        end
        if (isempty(gcp('nocreate')))
            parpool(n_workers);
        end
        parfor subject_idx = 1:length(unique_subject_ID)
%             waitbar((subject_idx-1)/length(unique_subject_ID),h,...
            fprintf(['Processing Subject #',num2str(subject_idx),' of ',num2str(length(unique_subject_ID)),'\n'])
            
            % load previous output; if the model has already been run for that
            % subject, skip it and save the previous output
            
%             if (~exist(band_data_base_filename,'dir'))
%                 mkdir(band_data_base_filename)
%             end
%             if (exist([band_data_base_filename,'/',workspace_output_name],'file')==2)
%                 previous_run_output = load([band_data_base_filename,'/',workspace_output_name]);
%                 flag_load_values = ismember(unique_subject_ID(subject_idx),previous_run_output.unique_subject_ID);
%             else
%                 flag_load_values = 0;
%             end
%             if (flag_load_values && unique_subject_ID(subject_idx)~=10 && unique_subject_ID(subject_idx)~=11)
%                 cv_svm_models = previous_run_output.svm_models{find(...
%                     unique_subject_ID(subject_idx)==previous_run_output.unique_subject_ID)};
%                 optimal_box_constraint = ...
%                     previous_run_output.optimal_box_constraint_all_subjects(find(...
%                     unique_subject_ID(subject_idx)==previous_run_output.unique_subject_ID));
%             else
                X = training_and_testing_data{subject_idx,1}; % matrix M (number of samples) by N (number of features) of feature data
                Y = training_and_testing_data{subject_idx,2}; % cell array M by 1 of labels
                [cv_svm_models,optimal_box_constraint,result] = build_SVM(X,Y,K_FOLD);
%             end
            %%% store results
            individual_svm_filename = sprintf('%s/PD_%02d.mat',SVM_dir,unique_subject_ID(subject_idx));
            parfor_save(individual_svm_filename,cv_svm_models,optimal_box_constraint,result)
            
            svm_models{subject_idx} = cv_svm_models;
            optimal_box_constraint_all_subjects(subject_idx) = optimal_box_constraint;
            bayesopt_results_all{subject_idx} = result;
        end
%         close(h)
        
        save([band_data_base_filename,'/',workspace_output_name]);
        
    end
end

function parfor_save(individual_svm_filename,cv_svm_models,optimal_box_constraint,result)
save(individual_svm_filename,'cv_svm_models','optimal_box_constraint','result');

