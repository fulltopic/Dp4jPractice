package dp4jpractice.org.mjdrl.nn.dqn.nn.replay

import org.nd4j.linalg.api.ndarray.INDArray

class LstmTransition(val rawInput: INDArray, val rawNextInput: INDArray, val qs: INDArray, val action: Integer, val reward: Double, val isTerminal: Boolean) {

}
