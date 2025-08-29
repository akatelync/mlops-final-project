To support my goal of predicting booking cancellations in the `ncr_ride_bookings.csv` dataset, I plan to introduce **covariate drift** to simulate real-world changes in ride-hailing patterns. This type of drift is ideal because it alters the distribution of input features (like `Vehicle Type` and `Payment Method`) without changing their relationship to cancellations, mimicking shifts in customer preferences that could challenge my model’s performance.

**Plan**: I’ll focus on two features:
- **Vehicle Type**: I’ll increase the proportion of `eBike/Bike` bookings from ~23% to 35% to reflect a trend toward eco-friendly vehicles, while reducing `UberXL` from ~3% to 1% due to lower demand in urban areas.
- **Payment Method**: I’ll shift toward digital payments, increasing `UPI` from ~40% to 50% and `Uber Wallet` from ~15% to 25%, while decreasing `Cash` from ~30% to 15%.

This drift simulates realistic market trends, like growing environmental awareness or cashless payment adoption, allowing me to test how well my cancellation prediction model adapts to new feature distributions. First, I’ll ensure the dataset loads correctly by addressing the parsing error (likely caused by unescaped commas in addresses) using robust CSV reading settings or by cleaning problematic rows.
