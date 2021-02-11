%{
X - matrix M (number of samples) by N (number of features) of feature
data

Y - cell array M (number of labels) by 1 of label data (i.e., values
must be 'off' or 'on')

K_FOLD - integer number of folds for cross validation

auto_optimize_box_constraint - optional flag; default value is 1; set
to 0 to specify a box constraint; otherwise, set to 1 to automatically
find an optimal box constraint value via BayesOpt

optimal_box_constraint - optional scalar value for box constraint;
must be specified if auto_optimize_box_constraint is set to zero

%}
function [cv_svm_models,optimal_box_constraint,result] = build_SVM(X,Y,K_FOLD,auto_optimize_box_constraint,optimal_box_constraint)

if (nargin<4)
    auto_optimize_box_constraint = 1;
end

if (nargin<5)
    assert(auto_optimize_box_constraint==1,'If auto_optimize_box_constraint is set to 0, you must specify a box constraint value...')
end
    
    
%%% manually calculate fold indices without shuffling (i.e., to ensure
%%% that test data is not mixed in with the training data)

X_off = X(strcmp('off',Y),:);
Y_off = Y(strcmp('off',Y));

X_on = X(strcmp('on',Y),:);
Y_on = Y(strcmp('on',Y));

cv_partition_off_and_on = cvpartition([Y_off;Y_on],'KFold',K_FOLD); 
% partition_indices = cv_partition_off_and_on.Impl.indices;

fold_indices = nan(length([Y_off;Y_on]),1);
test_data_indices_all_folds = arrayfun(@(x) find(test(cv_partition_off_and_on,x))',1:K_FOLD,'UniformOutput',false);

for fold_i = 1:K_FOLD
    fold_indices(test_data_indices_all_folds{fold_i}) = fold_i;
end
% assert(isequal(fold_indices,partition_indices))

fold_indices_off = fold_indices(strcmp('off',[Y_off;Y_on]));
[~,sort_order_off] = sort(fold_indices_off); [~,sort_order_off_squared]=sort(sort_order_off);
X_off_rearranged = X_off(sort_order_off_squared,:);
Y_off_rearranged = Y_off(sort_order_off_squared);

fold_indices_on = fold_indices(strcmp('on',[Y_off;Y_on]));
[~,sort_order_on] = sort(fold_indices_on); [~,sort_order_on_squared]=sort(sort_order_on);
X_on_rearranged = X_on(sort_order_on_squared,:);
Y_on_rearranged = Y_on(sort_order_on_squared);

% rearrange original data so that the fold indices are contiguous in
% time
X_rearranged = [X_off_rearranged; X_on_rearranged];
Y_rearranged = [Y_off_rearranged; Y_on_rearranged];


% create dummy cvpartition object
% cv_partition_off_and_on = cvpartition(size(Y,1),'KFold',K_FOLD);

% % use cvpartition to auto-generate partition sizes for the manually calculate 'off' condition fold incdices
% off_fold_divisions = cvpartition(sum(strcmp(Y,'off')),'KFold',K_FOLD);
% off_manual_fold_indices = arrayfun(@(fold_i,fold_partition_i) fold_i*ones(...
%     fold_partition_i,1),1:K_FOLD,off_fold_divisions.TestSize,'UniformOutput',false);
% off_manual_fold_indices = vertcat(off_manual_fold_indices{:});
% 
% % calculate the on condition fold incdices
% on_fold_divisions_target_test_size = cv_partition_off_and_on.TestSize - off_fold_divisions.TestSize;
% on_manual_fold_indices = arrayfun(@(fold_i,fold_partition_i) fold_i*ones(...
%     fold_partition_i,1),1:K_FOLD,on_fold_divisions_target_test_size,'UniformOutput',false);
% on_manual_fold_indices = vertcat(on_manual_fold_indices{:});
% 
% manual_fold_indices_off_and_on = zeros(size(Y));
% manual_fold_indices_off_and_on(strcmp(Y,'off')) = off_manual_fold_indices;
% manual_fold_indices_off_and_on(strcmp(Y,'on')) = on_manual_fold_indices;

% cv_partition_off_and_on.Impl.indices = manual_fold_indices_off_and_on;

%%% Use Bayesian Optimization to identify optimal box contraint
if (auto_optimize_box_constraint)
    box_constraint_optimizable_variable = optimizableVariable('box_constraint',[1e0,1e7],'Transform','log');
    minfn = @(z_two) kfoldLoss(fitcsvm(X_rearranged,Y_rearranged,...
        'CVPartition',cv_partition_off_and_on,...
        'KernelFunction','linear','BoxConstraint',z_two.box_constraint,...
        'KernelScale',1,'Standardize',false,'ClassNames',{'off','on'}));
    result = bayesopt(minfn,...
        box_constraint_optimizable_variable,'MaxObjectiveEvaluations',30,'Verbose',1,...
        'IsObjectiveDeterministic',false,'PlotFcn',[],... setting to false because objective is noisy (even though it actually is deterministic)
        'AcquisitionFunctionName','expected-improvement-plus');
    optimal_box_constraint = result.XAtMinObjective.box_constraint;
end


%%% Fit cross validated models using optimal box constraint
tic
cv_svm_models = fitcsvm(X_rearranged,Y_rearranged,...
    'CVPartition',cv_partition_off_and_on,...
    'KernelFunction','linear','BoxConstraint',optimal_box_constraint,...
    'KernelScale',1,'Standardize',false,'ClassNames',{'off','on'});
toc