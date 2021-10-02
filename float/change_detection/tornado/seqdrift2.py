from float.change_detection.base_change_detector import BaseChangeDetector
import math
import random
import sys


class SeqDrift2(BaseChangeDetector):
    """ SeqDrift2 Drift Detection Method

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Pears, Russel, Sripirakas Sakthithasan, and Yun Sing Koh. "Detecting concept change in dynamic data streams."
    Published in: Machine Learning 97.3 (2014): 259-293.
    URL: https://link.springer.com/article/10.1007/s10994-013-5433-9

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, delta=0.01, block_size=200, reset_after_drift=False):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            delta (float):
            block_size (int):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False

        self.DELTA = delta
        self.BLOCK_SIZE = block_size
        self.seq_drift2 = _SeqDrift2Tornado(self.DELTA, self.BLOCK_SIZE)

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.seq_drift2 = _SeqDrift2Tornado(self.DELTA, self.BLOCK_SIZE)

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = self.seq_drift2.setInput(pr)

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detected_partial_change(self):
        return False, None

    def detected_warning_zone(self):
        return False

    def get_length_estimation(self):
        pass


# ----------------------------------------
# Tornado Functionality (left unchanged)
# ----------------------------------------
class _SeqDrift2Tornado:
    """
    Tornado-class (SeqDrift2 renamed to _SeqDrift2Tornado, otherwise left unchanged)
    """
    def __init__(self, _significanceLevel, _blockSize):

        self.blockSize = _blockSize
        self.significanceLevel = _significanceLevel
        self.leftReservoirSize = _blockSize
        self.rightRepositorySize = _blockSize
        self.k = 0.5

        self.instanceCount = 0
        self.leftReservoirMean = 0
        self.rightRepositoryMean = 0
        self.variance = 0
        self.total = 0
        self.epsilon = 0

        self.DRIFT = 0
        self.NODRIFT = 2
        self.INTERNAL_DRIFT = 3

        self.rightRepository = _Reservoir(self.leftReservoirSize, self.blockSize)
        self.leftReservoir = _Reservoir(self.rightRepositorySize, self.blockSize)

    def setInput(self, _inputValue):
        self.instanceCount += 1
        self.addToRightReservoir(_inputValue)
        self.total += _inputValue

        if self.instanceCount % self.blockSize == 0:
            iDriftType = self.getDriftType()
            if iDriftType == self.DRIFT:
                self.clearLeftReservoir()
                self.moveFromRepositoryToReservoir()
                return True
            else:
                self.moveFromRepositoryToReservoir()
                return False

        return False

    def addToRightReservoir(self, _inputValue):
        self.rightRepository.addElement(_inputValue)

    def moveFromRepositoryToReservoir(self):
        self.leftReservoir.copy(self.rightRepository)

    def clearLeftReservoir(self):
        self.total -= self.leftReservoir.getTotal()
        self.leftReservoir.clear()

    def getDriftType(self):
        if self.getWidth() > self.blockSize:
            self.leftReservoirMean = self.getLeftReservoirMean()
            self.rightRepositoryMean = self.getRightRepositoryMean()
            self.optimizeEpsilon()

            if self.instanceCount > self.blockSize and self.leftReservoir.getSize() > 0:
                if self.epsilon <= abs(self.rightRepositoryMean - self.leftReservoirMean):
                    return self.DRIFT
                else:
                    return self.NODRIFT
            else:
                return self.NODRIFT
        else:
            return self.NODRIFT

    def getLeftReservoirMean(self):
        return self.leftReservoir.getSampleMean()

    def getRightRepositoryMean(self):
        return self.rightRepository.getSampleMean()

    def getVariance(self):
        mean = self.getMean()
        meanminum1 = mean - 1
        size = self.getWidth()
        x = self.getTotal() * meanminum1 * meanminum1 + (size - self.getTotal()) * mean * mean
        y = size - 1
        return x / y

    def optimizeEpsilon(self):
        tests = self.leftReservoir.getSize() / self.blockSize
        if tests >= 1:
            variance = self.getVariance()
            if variance == 0:
                variance = 0.0001

            ddeltadash = self.getDriftEpsilon(tests)
            x = math.log(4 / ddeltadash)
            ktemp = self.k

            IsNotOptimized = True
            while IsNotOptimized:
                squareRootValue = math.sqrt(x * x + 18 * self.rightRepositorySize * x * variance)
                previousStepEpsilon = (1.0 / (3 * self.rightRepositorySize * (1 - ktemp))) * (x + squareRootValue)
                ktemp = 3 * ktemp / 4
                currentStepEpsilon = (1.0 / (3 * self.rightRepositorySize * (1 - ktemp))) * (x + squareRootValue)

                if ((previousStepEpsilon - currentStepEpsilon) / previousStepEpsilon) < 0.0001:
                    IsNotOptimized = False

            ktemp = 4 * ktemp / 3
            ktemp = self.adjustForDataRate(ktemp)
            self.leftReservoirSize = int(self.rightRepositorySize * (1 - ktemp) / ktemp)
            self.leftReservoir.setMaxSize(self.leftReservoirSize)
            squareRootValue = math.sqrt(x * x + 18 * self.rightRepositorySize * x * variance)
            currentStepEpsilon = (1.0 / (3 * self.rightRepositorySize * (1 - ktemp))) * (x + squareRootValue)
            self.epsilon = currentStepEpsilon

    def getDriftEpsilon(self, _inumTests):
        dSeriesTotal = 2 * (1 - math.pow(0.5, _inumTests))
        ddeltadash = self.significanceLevel / dSeriesTotal
        return ddeltadash

    def getMean(self):
        return self.getTotal() / self.getWidth()

    def getTotal(self):
        return self.rightRepository.getTotal() + self.leftReservoir.getTotal()

    def adjustForDataRate(self, _dKr):
        meanIncrease = self.rightRepository.getSampleMean() - self.leftReservoir.getSampleMean()
        dk = _dKr
        if meanIncrease > 0:
            dk += ((-1) * (meanIncrease * meanIncrease * meanIncrease * meanIncrease) + 1) * _dKr
        elif meanIncrease <= 0:
            dk = _dKr
        return dk

    def getWidth(self):
        return self.leftReservoir.getSize() + self.rightRepository.getSize()

    def Estimation(self):
        iWidth = self.getWidth()
        if iWidth != 0:
            return self.getTotal() / self.getWidth()
        else:
            return 0

    def getDescription(self, sb, indent):
        pass


class _Reservoir:
    """
    Tornado-class (left unchanged)
    """
    def __init__(self, _iSize, _iBlockSize):
        self.size = 0
        self.total = 0
        self.blockSize = _iBlockSize
        self.dataContainer = _Repository(self.blockSize)
        self.instanceCount = 0
        self.MAX_SIZE = _iSize

    def getSampleMean(self):
        return self.total / self.size

    def addElement(self, _dValue):
        try:
            if self.size < self.MAX_SIZE:
                self.dataContainer.add(float(_dValue), None)
                self.total += _dValue
                self.size += 1
            else:
                irIndex = int(random.uniform(0, 1) * self.instanceCount)
                if irIndex < self.MAX_SIZE:
                    self.total -= self.dataContainer.get(irIndex)
                    self.dataContainer.addAt(irIndex, _dValue)
                    self.total += _dValue
            self.instanceCount += 1
        except ValueError:
            print("2 Expection", ValueError)

    def get(self, _iIndex):
        return self.dataContainer.get(_iIndex)

    def getSize(self):
        return self.size

    def clear(self):
        self.dataContainer.removeAll()
        self.total = 0
        self.size = 0

    def getTotal(self):
        return self.total

    def copy(self, _osource):
        for iIndex in range(0, _osource.getSize()):
            self.addElement(_osource.get(iIndex))
        _osource.clear()

    def setMaxSize(self, _iMaxSize):
        self.MAX_SIZE = _iMaxSize


class _Repository:
    """
    Tornado-class (left unchanged)
    """
    def __init__(self, _iBlockSize):
        self.blockSize = _iBlockSize
        self.blocks = []
        self.indexOfLastBlock = -1
        self.instanceCount = 0
        self.total = 0

    def add(self, _dValue, _isTested):
        if self.instanceCount % self.blockSize == 0:
            self.blocks.append(_Block(self.blockSize, _isTested))
            self.indexOfLastBlock += 1
        self.blocks[self.indexOfLastBlock].add(_dValue)
        self.instanceCount += 1
        self.total += _dValue

    def get(self, _iIndex):
        return self.blocks[int(_iIndex / self.blockSize)].data[(_iIndex % self.blockSize)]

    def addAt(self, _iIndex, _dValue):
        self.blocks[int(_iIndex / self.blockSize)].addAt((_iIndex % self.blockSize), _dValue)

    def getSize(self):
        return self.instanceCount

    def getTotal(self):
        dTotal = 0
        for iIndex in range(0, len(self.blocks)):
            dTotal += self.blocks[iIndex].total
        return dTotal

    def getFirstBlockTotal(self):
        return self.blocks[0].total

    def markLastAddedBlock(self):
        if len(self.blocks) > 0:
            self.blocks[len(self.blocks) - 1].setTested(True)

    def removeFirstBlock(self):
        self.total -= self.blocks[0].total
        self.blocks.pop(0)
        self.instanceCount -= self.blockSize
        self.indexOfLastBlock -= 1

    def removeAll(self):
        self.blocks.clear()
        self.indexOfLastBlock = -1
        self.instanceCount = 0
        self.total = 0

    def getNumOfTests(self):
        iNumTests = 0
        for iIndex in range(0, len(self.blocks)):
            if self.blocks[iIndex].IsTested():
                iNumTests += 1
        return iNumTests


class _Block:
    """
    Tornado-class (left unchanged)
    """
    def __init__(self, _iLength, _isTested=None):
        self.data = []
        self.total = 0
        self.indexOfLastValue = 0
        self.b_IsTested = _isTested

        for i in range(0, _iLength):
            self.data.append(-1)

    def add(self, _dValue):
        if self.indexOfLastValue < len(self.data):
            self.data[self.indexOfLastValue] = _dValue
            self.total += _dValue
            self.indexOfLastValue += 1
        else:
            print("Error in adding to Block. Last Index:", self.indexOfLastValue,
                  "Total", self.total, "Array Length:", len(self.data))
            sys.exit(2)

    def addAt(self, _iIndex, _dNewValue):
        self.total = self.total - self.data[_iIndex] + _dNewValue
        self.data[_iIndex] = _dNewValue

    def setTested(self, _isTested):
        self.b_IsTested = _isTested

    def IsTested(self):
        return self.b_IsTested
