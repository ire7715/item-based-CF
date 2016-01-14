import numpy as np
import sys
import time
import os.path

def cosSimilarity(v1, v2):
  return v1.dot(v2) / np.sqrt(np.square(v1).sum()) / np.sqrt(np.square(v2).sum())

def amazonSimilarity(reviews, userIndex, itemIndex, ratingIndex):
  dims = np.amax(reviews, axis=0)
  maxUser = int(dims[userIndex] + 1)
  maxItem = int(dims[itemIndex] + 1)

  i2uMatrix = np.zeros(shape=(maxItem, maxUser))
  u2iMatrix = np.zeros(shape=(maxUser, maxItem))
  for review in reviews:
    thisUser = int(review[userIndex])
    thisItem = int(review[itemIndex])
    i2uMatrix[thisItem][thisUser] = review[ratingIndex]
    u2iMatrix[thisUser][thisItem] = review[ratingIndex]
  itemsOrderedBy = u2iMatrix > 0
  usersPurchased = i2uMatrix > 0

  print("preprocessing is done, calculating similarity...")

  similarities = np.zeros(shape=(maxItem, maxItem))
  for i in range(maxItem):
    relatedItems = itemsOrderedBy[usersPurchased[i]].sum() > 0
    filteredU2i = u2iMatrix * relatedItems
    similarities[i] = np.apply_along_axis(cosSimilarity, 0, filteredU2i, i2uMatrix[i])

  print
  return similarities, u2iMatrix, itemsOrderedBy, usersPurchased

def predict(itemsSimilarity, userRatingOnItems):
  EPS = 1e-7
  score = np.dot(itemsSimilarity, userRatingOnItems)
  sim1Norm = np.linalg.norm(itemsSimilarity, ord=1)
  if sim1Norm < EPS:
    score = 0
  else:
    score = score / sim1Norm
  if score < 0:
    score = 0
  elif score > 5:
    score = 5
  return np.round(score)

def trainedModelsExist(trainLabel):
  models = [".amazon_similarity", ".userMajoredMatrix", ".itemsOrderedBy", ".usersPurchased"]
  allExist = True
  for model in models:
    if not os.path.isfile(trainLabel + model + ".npy"):
      allExist = False
      break
  return allExist

def readTrainedModels(trainLabel):
  similarities = np.load(trainLabel + ".amazon_similarity.npy")
  userMajoredMatrix = np.load(trainLabel + ".userMajoredMatrix.npy")
  itemsOrderedBy = np.load(trainLabel + ".itemsOrderedBy.npy")
  usersPurchased = np.load(trainLabel + ".usersPurchased.npy")
  return similarities, userMajoredMatrix, itemsOrderedBy, usersPurchased

def main():
  if len(sys.argv) < 3:
    print("python amazon_i2i.py delimiter training_file [testing_file]")
    return 0
  delimiter = sys.argv[1]
  trainFile = sys.argv[2]
  trainLabel = trainFile[0:trainFile.rfind(".")]

  if trainedModelsExist(trainLabel):
    print("reading files...")
    similarities, userMajoredMatrix, itemsOrderedBy, usersPurchased = readTrainedModels(trainLabel)
    print("files read.")
  else:
    # user, item, rating, time
    reviews = np.genfromtxt(trainFile, delimiter=delimiter)

    print("similarity calculation started at " + time.strftime("%H:%M:%S"))
    similarities, userMajoredMatrix, itemsOrderedBy, usersPurchased = amazonSimilarity(reviews=reviews, userIndex=0, itemIndex=1, ratingIndex=2)
    reviews = None

    print("simiarity calculation ended at " + time.strftime("%H:%M:%S"))
    np.save(trainLabel + ".amazon_similarity", similarities)
    np.save(trainLabel + ".userMajoredMatrix", userMajoredMatrix)
    np.save(trainLabel + ".itemsOrderedBy", itemsOrderedBy)
    np.save(trainLabel + ".usersPurchased", usersPurchased)
    print("file outputed at " + time.strftime("%H:%M:%S"))

  if len(sys.argv) < 4:
    return 0
  testFile = sys.argv[3]
  testReviews = np.genfromtxt(testFile, delimiter=delimiter)

  print("prediction started at " + time.strftime("%H:%M:%S"))
  itemMajoredMatrix = None
  userMean = np.zeros(len(userMajoredMatrix))
  itemMean = np.zeros(len(userMajoredMatrix[0]))
  globalMean = None
  mse = 0
  confusionMatrix = np.zeros(shape=(2, 2), dtype=np.int16)
  scoreDistribution = [0, 0, 0, 0, 0, 0]
  for review in testReviews:
    user = int(review[0])
    item = int(review[1])
    actualScore = int(review[2])
    try:
      score = predict(similarities[item], userMajoredMatrix[user])
    except:
      if user >= len(userMajoredMatrix) and item < len(similarities):
        # unknown user
        if userMean[user] <= 0:
          userMean[user] = np.round(np.mean(np.extract(userMajoredMatrix[user] > 0, userMajoredMatrix[user])))
        score = userMean[user]
      elif item >= len(similarities) and user < len(userMajoredMatrix):
        # unknown item
        if itemMean[item] <= 0:
          if itemMajoredMatrix is None: itemMajoredMatrix = np.transpose(userMajoredMatrix)
          itemMean[item] = np.round(np.mean(np.extract(itemMajoredMatrix[item] > 0, itemMajoredMatrix[item])))
        score = itemMean[item]
      else:
        if globalMean is None:
          globalMean = np.round(np.mean(np.extract(userMajoredMatrix > 0, userMajoredMatrix)))
        score = globalMean
    if score < 1:
      if user < len(userMean):
        if userMean[user] <= 0:
          userMean[user] = np.round(np.mean(np.extract(userMajoredMatrix[user] > 0, userMajoredMatrix[user])))
        score = userMean[user]
      elif item < len(itemMean):
        if itemMean[item] <= 0:
          if itemMajoredMatrix is None: itemMajoredMatrix = np.transpose(userMajoredMatrix)
          itemMean[item] = np.round(np.mean(np.extract(itemMajoredMatrix[item] > 0, itemMajoredMatrix[item])))
        score = itemMean[item]
      else:
        if globalMean is None:
          globalMean = np.round(np.mean(np.extract(userMajoredMatrix > 0, userMajoredMatrix)))
        score = globalMean
    mse += (actualScore - score) ** 2
    confusionMatrix[actualScore >= 3][score >= 3] += 1
    scoreDistribution[int(score)] += 1
  mse /= len(testReviews)
  print("MSE: " + str(mse))
  print("Confusion Matrix: ")
  print(confusionMatrix)
  print("Score distribution: ")
  print(scoreDistribution)
  print("prediction ended at " + time.strftime("%H:%M:%S"))

  return 0

if __name__ == "__main__":
  main()
