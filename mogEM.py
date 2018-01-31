from kmeans import *
import sys
import matplotlib.pyplot as plt
#plt.ion()

def mogEM(x, K, iters, minVary=0.0, use_Kmeans=False):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  if use_Kmeans:
    mu = KMeans(x,K,5)
  else:
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  
  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM without K means initialization')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def train_MoG():
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz',load2=False)
  maxProb = -float("inf")
  max_mu = []
  max_vary = []
  max_p = []
  max_log = []
  for i in range(10):
    p, mu, vary, logProbX = mogEM(inputs_train,2,iters,minVary = 0.01)
    if logProbX[9] > maxProb:
      maxProb = logProbX[9]
      max_log = logProbX
      max_mu = mu
      max_vary = vary
      max_p = p
  print("max prob: " + str(maxProb))
  print("mixing proportions" + str(max_p))
  #plt.plot(max_log)
  plt.show()
  ShowMeans(max_mu)
  ShowMeans(max_vary)

def initialize_MoG_Kmeans():
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  

  p, mu, vary, logProbX = mogEM(inputs_train,20,iters,minVary = 0.01,use_Kmeans=False)

  print("max prob: " + str(logProbX))
  print("mixing proportions" + str(p))
  #plt.plot(max_log)
  plt.show()
  ShowMeans(mu)
  ShowMeans(vary)

  raw_input('Press Enter to continue.')
def classify():
  iters = 10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  #print(errorTrain)
  numComponents = np.array(range(2,26))
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  #print(target_test2)
  error_train = []
  error_valid = []
  error_test = []
  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    p1, mu1, vary1, logProbX1 = mogEM(train2,K,iters,minVary = 0.01,use_Kmeans=True)
    #ShowMeans(mu1)

    
    # Train a MoG model with K components for digit 3
    p2, mu2, vary2, logProbX2 = mogEM(train3,K,iters,minVary = 0.01,use_Kmeans=True)
    #ShowMeans(mu2)

    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    prob1 = mogLogProb(p1,mu1,vary1,train2)
    prob2 = mogLogProb(p2,mu2,vary2,train2)
    incorrect = get_pct_incorrect(prob1,prob2,2)
    prob1 = mogLogProb(p1,mu1,vary1,train3)
    prob2 = mogLogProb(p2,mu2,vary2,train3)
    incorrect = (incorrect + get_pct_incorrect(prob1,prob2,3))/2.0
    error_train.append(incorrect)

    prob1 = mogLogProb(p1,mu1,vary1,test2)
    prob2 = mogLogProb(p2,mu2,vary2,test2)
    incorrect = get_pct_incorrect(prob1,prob2,2)
    prob1 = mogLogProb(p1,mu1,vary1,test3)
    prob2 = mogLogProb(p2,mu2,vary2,test3)
    incorrect = (incorrect + get_pct_incorrect(prob1,prob2,3))/2.0
    error_test.append(incorrect)

    prob1 = mogLogProb(p1,mu1,vary1,valid2)
    prob2 = mogLogProb(p2,mu2,vary2,valid2)
    incorrect = get_pct_incorrect(prob1,prob2,2)
    prob1 = mogLogProb(p1,mu1,vary1,valid3)
    prob2 = mogLogProb(p2,mu2,vary2,valid3)
    incorrect = (incorrect + get_pct_incorrect(prob1,prob2,3))/2.0
    error_valid.append(incorrect)


    
  # Plot the error rate
  plt.clf()
  print(np.shape(numComponents))
  print(np.shape(error_train))
  plt.plot(numComponents,error_train,label="training")
  plt.plot(numComponents,error_valid,label='valid')
  plt.plot(numComponents,error_test,label='test')
  plt.title("Average classification errors vs number of mixture components")
  plt.legend()
  plt.show()
  #raw_input('Press Enter to continue.')
def get_pct_incorrect(prob1,prob2,actual):
  guess = []
  for i in range(len(prob1)):
    if prob1[i] > prob2[i]:
      guess.append(2)
    else:
      guess.append(3)
  num_incorrect = 0
  for i in range(len(guess)):
    if guess[i] != actual:
      num_incorrect += 1
  return num_incorrect/float(len(guess))



