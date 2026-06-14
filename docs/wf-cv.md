# Walk-Forward Cross Validation

## The concept
walk forward is a technique assuring the optimization method works for different time frames, and market regimes, so that, if applied periodically, trading strategy parameters can be optimized based on the most recent changes in the market. It also allows for continuing validation of the optimization method.

## Implementation
1. Select the minimally acceptable trainding interval in number of selected data intervals (e.g 1m, 1hr, 1d)
2. Select adjuscent validation interval (approx. 1/3 of training period)
3. Select adjuscent test interval (equal to validation period) aka OOS (out of sample)
4. Run optimization on the training interval and achieve conversion 
5. Validate the found optimized parameters on validation interval (the performance normally must be about the same). If significant difference is occured, adjust training parameters to avoid overfitting and run training again (p. 4)
6. Test and record in OOS
7. After the OOS test is done, shift training period to the later data points. Normally the shift equals the length of the validation period. This way OOS windows will be adjuscent and constitute continuous space of OOS data after the WF completion
8. Repeat the previous steps till the end of available data
9. Report OOS performance

### Minimally accepted training interval 
The interval length depends on the number of available data points that are used in the training and number of parameters to be optimized. The rule of thumb: the number of data points = number of parameters x 10 