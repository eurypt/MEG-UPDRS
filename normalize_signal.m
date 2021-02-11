%{

Description: 
Returns the normalized signal, 'demeaned_unit_variance_signal', from the 
original signal, 'signal'. The normalized signal has zero mean and unit variance.

Input Arguments:
signal (vector | matrix) - Input signal, specified as either a row vector
or column vector, or a matrix. When 'signal' is a matrix, the columns are
processed as separate signals

%}
function demeaned_unit_variance_signal = normalize_signal(signal)

demeaned_signal = bsxfun(@minus,signal,mean(signal));
demeaned_unit_variance_signal = bsxfun(@rdivide,demeaned_signal,std(demeaned_signal));

% Check mean and variance of new signal
assert(all(abs(mean(demeaned_unit_variance_signal)./std(demeaned_unit_variance_signal))<1e-10),'Error: Normalized signal must have zero mean')
assert(all(round(var(demeaned_unit_variance_signal),5)==1),'Error: Normalized signal must have unit variance')

end