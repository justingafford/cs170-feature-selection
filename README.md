# cs170-feature-selection

CustomSearch:
We do a nearest neighbor on all our features, then add them to a priority queue with the priority: (1 - accuracy), this ensures that our features are sorted by their accuracy. We then sort through our possible features and select the best feature to our set, adding our features from our queue. If we have a decrease in accuracy due to a feature we go through more features in our heap based upon our depth, which is dependent on our number of features. We then check to see if there's any valid feature, if not, we can then reinitialize our heap. The runtime of our algorithm increases with everytime we reinitialize our heap, so we have to minimize the times we do this. If our reinitization improves our accuracy, we replace our accuracy the new accuracy.
