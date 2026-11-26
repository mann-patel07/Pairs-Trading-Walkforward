# Plan for the rebalancing cost check

Date: 2026-11-26

Written before these results existed. The pre-registered analysis and its numbers are unchanged.

## Question
The portfolio holds an equal weight in each eligible pair every day. Keeping the weights equal means moving money between pairs each day, and that trading is not costed in the first version. How much does it cost?

## Method
- Daily cost on day t in window k, with n_k eligible pairs:
  cost_t = (COST_BPS / 10000) x (1 / n_k) x sum over eligible pairs i with an open position on day t of abs(1 - (1 + r_i,t) / (1 + r_p,t)),
  where r_i,t is pair i's net return, r_p,t is the portfolio's net return on day t, and COST_BPS = 5.
- Pairs with no open position hold cash, and moving cash costs nothing.
- Report net Sharpe ratio, annual return and maximum drawdown for ALL, COINT and COINT_BH with and without this cost, and the average daily cost.
- Sensitivity only. It does not replace the headline numbers.
