package rl.dqn.supervised.fileprocess

import akka.event.slf4j.Logger
import org.deeplearning4j.rl4j.learning.sync.Transition
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import rl.dqn.reinforcement.dqn.config.Supervised._

import scala.xml.XML
import scala.xml.Node
import scala.xml.Text
//import scala.collection.immutable.Vector

//TODO: Remove scene with BYE

class FileProcessor (fileName: String) {
  private val logger = Logger("FileProcessor")
  
  var trans = scala.collection.immutable.Vector.empty[Transition[Int]]
  val currentStats: Array[Array[Int]] = new Array[Array[Int]](PlayerNum)
//    Nd4j.zeros(PlayerNum, StateLen)
  val tens = new Array[Int](PlayerNum)
  var oya: Int = 0
//  private[this] val isReach = new Array[Boolean](PlayerNum)
  val isReach = Array.fill[Boolean](PlayerNum)(false)
  val doraValue = Array.fill[Int](StateLen)(0)
//  private[this] val doraValue = new Array[Int](StateLen)
  var gameCount: Int = 0

  for(i <- currentStats.indices) {
    currentStats(i) = new Array[Int](StateLen)
  }

  def readFile(): Unit = {
    val root = XML.loadFile(fileName)
    val children = root.descendant
    logger.info(children.size + "")
    val a = root \ "Scene"

    a.foreach(readScene(_))

//    readScene(a.head)
  }

  def readScene(sceneNode: Node): Unit = {
    gameCount += 1
    logger.info("Read game " + gameCount)
    val byeNode = sceneNode \ "BYE"
    if (byeNode.length > 0) {
      logger.info("--------------------------> Interrupted game, give up")
    }else{
      sceneNode.child.foreach { child => {
        child match {
          case _: Text => //logger.info(child.head.label)
          case _: Node => readNodeInSeq(child)
        }
      }
      }
    }
  }

  def readNodeInSeq(node: Node): Unit = {
//    logger.info("===========================> Get a node " + node.label)
    val label = node.label

    label match {
      case "INIT" => parseInit(node)
      case "N" => parseSteal(node)
      case "AGARI" => parseTerminal(node)
      case "RYUUKYOKU" => parseTerminal(node)
      case "REACH" => parseReach(node)
      case "DORA" => parseDora(node)
      case _ => parsePlayer(node)
    }
  }

  //TODO: Initial dora
  def parseInit(node: Node): Unit = {
//    logger.info("init ")
//    logger.info(node.attributes.size)

    // ten
    val ten = node.attribute("ten")
    ten match {
      case Some(x) => {
        val initTens = x.head.text.split(",").map(_.toInt)
        for (i <- 0 until PlayerNum) {
          tens(i) = initTens(i)
        }
      }
      case _ => //do nothing
    }

    // oya
    oya = node.attribute("oya").map(_.head.text.toInt).getOrElse(0)
    currentStats(oya)(OyaIndex) = 1
//    logger.info(oya)

    // dora
    val dora = node.attribute("seed").map(_.head.text.split(",").map(_.toInt).apply(DoraInSeed)).getOrElse(0)
    parseDora(dora / NumPerTile)
//    val seedAttr = node.attribute("seed")
//    seedAttr match {
//      case Some(x) => {
//        val seeds = x(0).text.split(",").map(_.toInt)
//        parseDora(seeds(seeds.length - 1) / NumPerTile)
//      }
//      case _ => //nothing
//    }

    val hai0 = node.attribute("hai0")
    parseInitPlayer(hai0, 0)
    val hai1 = node.attribute("hai1")
    parseInitPlayer(hai1, 1)
    val hai2 = node.attribute("hai2")
    parseInitPlayer(hai2, 2)
    val hai3 = node.attribute("hai3")
    parseInitPlayer(hai3, 3)

//    currentStats.foreach(state => {
//      state.foreach(print)
//      logger.info("")
//    })
  }

  def parseInitPlayer(hai: Option[Seq[Node]], index: Int): Unit = {
    hai match {
      case Some(x) => {
        x.head.text.split(",").map(_.toInt).foreach(i => {
            val tile = i / NumPerTile
            currentStats(index)(tile) = currentStats(index)(tile) + 1 + doraValue(tile)
            if (isAkaDora(i)) { currentStats(index)(tile) += DoraValue}
        })}
      case _ => //nothing
    }
  }

  def parsePlayer(node: Node): Unit = {
//    logger.info("player " + node.text)
    node.label.charAt(0) match {
      case 'T' => parseAccept(node, 0)
      case 'D' => parseDrop(node, 0)
      case 'U' => parseAccept(node, 1)
      case 'E' => parseDrop(node, 1)
      case 'V' => parseAccept(node, 2)
      case 'F' => parseDrop(node, 2)
      case 'W' => parseAccept(node, 3)
      case 'G' => parseDrop(node, 3)
      case _ => logger.error("------------------> Received unexpected node " + node.label)
    }
  }

  def createTransition(state: Array[Int], nextState: Array[Int], action: Int, reward: Int, gameOver: Boolean): Unit = {
    val (stateIND, nextStateIND) = createINDState(state, nextState)
    val transition = new Transition[Int](stateIND, action, reward, gameOver, nextStateIND)
    trans = trans :+ transition
  }

  def isAkaDora(hai: Int): Boolean = {
    hai == 16 || hai == 52 || hai == 88
  }

  def parseAccept(node: Node, index: Int): Unit = {
//    if (!isReach(index)) {
      val origTile = node.label.substring(1).toInt
      val tile = origTile / NumPerTile
      val currState = currentStats(index).clone()

      currentStats(index)(tile) = currentStats(index)(tile) + doraValue(tile) + 1
      if (isAkaDora(origTile)) { currentStats(index)(tile) += DoraValue }

//      createTransition(currState, currentStats(index), Accept, DefaultReward, false)
//    }
  }

  def parseDrop(node: Node, index: Int): Unit = {
//    if (!isReach(index)) {
      val origTile = node.label.substring(1).toInt
      val tile = origTile / NumPerTile
      val currState = currentStats(index).clone()
      currentStats(index)(tile) -= (1 + doraValue(tile))
      if (isAkaDora(origTile)) {currentStats(index)(tile) -= DoraValue}

      createTransition(currState, currentStats(index), tile, DefaultReward, false)
//      logger.info("Drop " + index + " = " + tile)
//    }
  }

  // To give action except chow and pong as it is seemed little chance to happen
  def parseSteal(node: Node): Unit = {
//    logger.info("steal " + node.text)
    val whoAttr = node.attribute("who")
    whoAttr match {
      case Some(x) => {
        val who = x.head.text.toInt

        val mAttr = node.attribute("m")
        mAttr match {
          case Some(y) => {
            val m = y.text.toInt

            if ((m & ChowFlag) > 0) {
              parseChow(who, m)
            }else if ((m & PongFlag) > 0) {
              parsePong(who, m)
            }else if ((m & KakanFlag) > 0) {
              parseKakan(who, m)
            }else if ((m & AnkanFlag) == 0) {
              parseAnkan(who, m)
            }else if ((m >> KitaBits & 1) == 1) {
              parseKita(who, m)
            }else {
              parseMinkan(who, m)
            }
          }
          case _ => //nothing
        }
      }
      case _ => //nothing
    }
  }

  def parseKita(index: Int, m: Int): Unit = {
    //nothing
  }

  def parseMinkan(index: Int, m: Int): Unit = {
    var kanTile = (m >> 8) & 255
    val state = currentStats(index).clone()
    currentStats(index)(kanTile) += 1
    createTransition(state, currentStats(index), MinKan, DefaultReward, false)
  }


  def parseAnkan(index: Int, m: Int): Unit = {
    var ankanTile = (m >> 8) & 255
    val isAka = isAkaDora(ankanTile)

    ankanTile = ankanTile / 4

    val currentState = currentStats(index).clone()
    val origValue = currentStats(index)(ankanTile)
    currentStats(index)(ankanTile) = origValue & ExtraValueFlag + MValue * 4 + doraValue(ankanTile)
    if (isAka) {currentStats(index)(ankanTile) += DoraValue}
    createTransition(currentState, currentStats(index), AnKan, DefaultReward, false)
  }

  // Should tile / 4?
  def parseKakan(index: Int, m: Int): Unit = {
    val unused = (m >> 5) & 3
    var kakanTile = (m >> 9) & 127

    kakanTile /= 3
    kakanTile *= 4
    kakanTile += unused
    val isAka = isAkaDora(kakanTile)
    kakanTile /= NumPerTile

    val currentState = currentStats(index).clone()
    val origValue = currentStats(index)(kakanTile)
    currentStats(index)(kakanTile) = origValue & ExtraValueFlag + MValue * 4 + doraValue(kakanTile)
    if (isAka) { currentStats(index)(kakanTile) += DoraValue}
    createTransition(currentState, currentStats(index), KaKan, DefaultReward, false)
  }

  def parsePong(index: Int, m: Int): Unit = {
    var pongTile = (m >> 9) & 127
    val r = pongTile % 3
    pongTile /= 3
    pongTile *= 4
    pongTile += r
    pongTile /= NumPerTile

//    logger.info("=====......................... Get pong tile " + pongTile)
    val currentState = currentStats(index).clone()
//    currentStats(index)(pongTile) += (1 + doraValue(pongTile))
    currentStats(index)(pongTile) += (doraValue(pongTile) + 1 - 3 + MValue * 3)
    createTransition(currentState, currentStats(index), Pong, DefaultReward, false)
  }

  def parseChow(index: Int, m: Int): Unit = {
    var chowTile = (m >> 10) & 63
    val r = chowTile % 3

    chowTile /= 3
    chowTile = chowTile / 7 * 9 + chowTile % 7
    chowTile *= 4

    val candidates = new Array[Int](3)
    candidates(0) = chowTile + ((m >> 3) & 3)
    candidates(1) = chowTile + 4 + ((m >> 5) & 3)
    candidates(2) = chowTile + 8 + ((m >> 7) & 3)

    val peers = new Array[Int](2)
    r match {
      case 0 =>
        chowTile = candidates(0)
        peers(0) = candidates(1)
        peers(1) = candidates(2)
      case 1 =>
        chowTile = candidates(1)
        peers(0) = candidates(0)
        peers(1) = candidates(2)
      case 2 =>
        chowTile = candidates(2)
        peers(0) = candidates(0)
        peers(1) = candidates(1)
    }
//    r match {
//      case 0 => chowTile = chowTile + ((m >> 3) & 3)
//      case 1 => chowTile = chowTile + 4 + ((m >> 5) & 3)
//      case 2 => chowTile = chowTile + 8 + ((m >> 7) & 3)
//    }

    chowTile /= NumPerTile
    val currentState = currentStats(index).clone()
//    currentStats(index)(chowTile) += (1 + doraValue(chowTile))
    currentStats(index)(chowTile) += (MValue + doraValue(chowTile))
    peers.foreach(f = tile => {
      currentStats(index)(tile / 4) += (-1 + MValue)
    })
    createTransition(currentState, currentStats(index), Chow, DefaultReward, false)
  }

  def createINDState(state: Array[Int], nextState: Array[Int]): (Array[INDArray], INDArray) = {
    val stateIND = new Array[INDArray](1)
    stateIND(0) = Nd4j.zeros(1, StateLen)
    val nextStateIND = Nd4j.zeros(1, StateLen)

    for(i <- 0 until StateLen) {
      stateIND(0).putScalar(i, state(i))
      nextStateIND.putScalar(i, nextState(i))
    }

    (stateIND, nextStateIND)
  }

  def getUpdateStateValue(tile: Int): Int = {
    var v = 1 + doraValue(tile / NumPerTile)
    if(isAkaDora(tile)) {
      v += DoraValue
    }

    v
  }

  def parseTerminal(node: Node): Unit = {
    logger.info("end of game " + gameCount)

    val rewards = node.attribute("sc").map(_.text.split(",").map(_.toInt).zipWithIndex.filter(_._2 % 2 == 1).map(_._1))
                    .getOrElse(Array.fill[Int](PlayerNum)(0))
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(-1)
    for (i <- rewards.indices) {
      if (i == who) {
        val machi = node.attribute("machi").map(_.head.text.toInt).getOrElse(0)
        val state = currentStats(i).clone()
        currentStats(i)(machi / NumPerTile) += getUpdateStateValue(machi)
        createTransition(state, currentStats(i), Ron, rewards(i), true)
      }else {
        createTransition(currentStats(i), currentStats(i), NOOP, rewards(i), true)
      }
    }
  }

  def parseReach(node: Node): Unit = {
    val step = node.attribute("step").map(_.head.text.toInt).getOrElse(0)
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(0)
    step match {
      case 1 =>
        val currentState = currentStats(who).clone()
        currentStats(who)(ReachIndex) = ReachStep1
        createTransition(currentState, currentStats(who), REACH, ReachReward, false)
//        logger.info("---------------------------> Reach " + who)
      case 2 =>
        //TODO: Suppose REACH step = 2 will be sent by server
        isReach(who) = true
        val currentState = currentStats(who).clone()
        currentStats(who)(ReachIndex) = ReachStep2
        createTransition(currentState, currentStats(who), NOOP, DefaultReward, false)

      case _ => logger.error("Received invalid reach step " + step)
    }
  }

  def getDora(hai: Int): Int = {
    hai match {
      case tile if tile >= 0 && tile < 27 =>
        val tmp = tile / 9
        (tile + 1) % 9 + tmp * 9
      case tile if tile >= 27 && tile < 31 =>
        val tmp = tile - 27
        (tmp + 1) % 4 + 27
      case _ =>
        val tmp = hai - 31
        (hai + 1) % 3 + 31
    }
  }

  def parseDora(hai: Int): Unit = {
    val doraHai = getDora(hai)
    doraValue(doraHai) = DoraValue
    currentStats.foreach(state => {
      if (state(doraHai) > 0) {
        val tile = state(doraHai) % DoraValue
//        state(doraHai) = (state(doraHai) / DoraValue) * DoraValue + tile * DoraValue
        state(doraHai) += tile * DoraValue
      }
    })
  }

  def parseDora(node: Node): Unit = {
    val hai = node.attribute("hai").map(_.head.text.toInt / NumPerTile).getOrElse(0)
    val origStates = new Array[Array[Int]](PlayerNum)
    for (i <- origStates.indices) {
      origStates(i) = currentStats(i).clone()
    }

    parseDora(hai)

    for(i <- origStates.indices) {
      createTransition(origStates(i), currentStats(i), NOOP, DefaultReward, false)
    }
  }

}
