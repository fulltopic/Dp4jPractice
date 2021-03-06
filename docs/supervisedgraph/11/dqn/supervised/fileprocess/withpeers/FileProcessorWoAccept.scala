package rl.dqn.supervised.fileprocess.withpeers

import akka.event.slf4j.Logger
import org.deeplearning4j.rl4j.learning.sync.Transition
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import rl.dqn.reinforcement.dqn.config.Supervised._
import rl.dqn.supervised.utils.ReplayDB

import scala.xml.{Node, Text, XML}

class FileProcessorWoAccept(fileName: String) {
  private val logger = Logger("FileProcessorWoAccept")

  var trans = scala.collection.immutable.Vector.empty[Transition[Integer]]
  val currentStats: Array[Array[Int]] = new Array[Array[Int]](PlayerNum)
  val tens = new Array[Int](PlayerNum)
  var oya: Int = 0
  val doraValue = Array.fill[Int](PeerStateLen)(0)
  private[this] var gameCount: Int = 0

  for(i <- currentStats.indices) {
    currentStats(i) = Array.fill[Int](PeerStateLen)(0)
  }

  def readFile(): Unit = {
    //    logger.info("-------------------------> To read file " + fileName)
    val root = XML.loadFile(fileName)
    val a = root \ "Scene"

    //    logger.info("Get game size " + a.length)

    a.foreach(readScene)
  }

  def readScene(sceneNode: Node): Unit = {
    gameCount += 1
    //    logger.info("Read game " + gameCount)
    val byeNode = sceneNode \ "BYE"
    if (byeNode.length > 0) {
      //      logger.info("--------------------------> Interrupted game, give up")
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
    node.label match {
      case "INIT" => parseInit(node)
      case "N" => parseSteal(node)
      case "AGARI" => parseTerminal(node)
      case "RYUUKYOKU" => parseTerminal(node)
      case "REACH" => parseReach(node)
      case "DORA" => parseDora(node)
      case "UN" => parseUN(node)
      case _ => parsePlayer(node)
    }
  }

  def parseUN(node: Node): Unit = {
    logger.error("Don't know what to do with UN")
  }

  //TODO: Initial dora
  def parseInit(node: Node): Unit = {
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
    currentStats(oya)(OyaOffset) = 1
    //    logger.info(oya)

    // dora
    val dora = node.attribute("seed").map(_.head.text.split(",").map(_.toInt).apply(DoraInSeed)).getOrElse(-1)
    parseDora(dora / NumPerTile)

    val hai0 = node.attribute("hai0")
    parseInitPlayer(hai0, 0)
    val hai1 = node.attribute("hai1")
    parseInitPlayer(hai1, 1)
    val hai2 = node.attribute("hai2")
    parseInitPlayer(hai2, 2)
    val hai3 = node.attribute("hai3")
    parseInitPlayer(hai3, 3)
  }

  def parseInitPlayer(hai: Option[Seq[Node]], index: Int): Unit = {
    hai match {
      case Some(x) => {
        x.head.text.split(",").map(_.toInt).foreach(i => {
          acceptTile(currentStats(index), i)
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

  private def createTransition(state: Array[Int], nextState: Array[Int], action: Int, reward: Int, gameOver: Boolean): Unit = {
    val (stateIND, nextStateIND) = createINDState(state, nextState)
    val transition = new Transition[Integer](stateIND, action, reward, gameOver, nextStateIND)
    trans = trans :+ transition
  }


  def parseAccept(node: Node, index: Int): Unit = {
    val origTile = node.label.substring(1).toInt
    val currState = currentStats(index).clone()
    acceptTile(currentStats(index), origTile)
//    createTransition(currState, currentStats(index), Accept, DefaultReward, false)
  }

  // tile = original value
  def acceptTile(state: Array[Int], tile: Int): Unit = {
    state(tile / NumPerTile) += 1 + doraValue(tile / NumPerTile) + AkaValues.getOrElse(tile, 0)
  }

  def dropTile(state: Array[Int], tile: Int): Unit = {
    state(tile / NumPerTile) -= 1 + doraValue(tile / NumPerTile) + AkaValues.getOrElse(tile, 0)
  }

  def fixTile(state: Array[Int], tile: Int): Unit = {
    state(tile / NumPerTile) += MValue - 1
  }

  def updatePlayer(index: Int, tiles: Array[Int], f: (Array[Int], Int) => Unit): Unit = {
    tiles.foreach(tile => f(currentStats(index), tile))
  }

  def updateBoard(tile: Int): Unit = {
    currentStats.foreach(state => {
      state(tile / NumPerTile + TileNum) += 1
    })
  }

  def updateBoard(tiles: Array[Int]): Unit = {
    currentStats.foreach(state => {
      tiles.foreach(tile => state(tile / NumPerTile + TileNum) += 1)
    })
  }

  //For other's action, no transition created, just update currentstate
  def parseDrop(node: Node, index: Int): Unit = {
    val tile = node.label.substring(1).toInt
    val origState = currentStats(index).clone()

    dropTile(currentStats(index), tile)
    updateBoard(tile)

    createTransition(origState, currentStats(index), tile / NumPerTile, DefaultReward, false)
  }

  // To give up actions except chow and pong as it is seemed little chance to happen
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

  //TODO: Implement kita for mjlog_pf4-20_n1/2009070621gm-0061-0000-841d1131&tw=3.xml
  // mjlog_pf4-20_n2/2010033123gm-0029-0000-53c6c5e8&tw=0.xml
  def parseKita(index: Int, m: Int): Unit = {
    //To throw exception
    logger.info("Want to process kita: " + m)
    logger.error("----------------------------------------> Kita has not been implemented " + currentStats(PlayerNum)(0))
  }

  def parseMinkan(index: Int, m: Int): Unit = {
    //exception would be thrown
    val kanTile = (m >> 8) & 255
    val state = currentStats(index).clone()

    acceptTile(currentStats(index), kanTile)
    val kanTiles = Array.fill[Int](NumPerTile)(kanTile)
    updatePlayer(index, kanTiles, fixTile)
    updateBoard(kanTiles.tail) //had one tile dropped before minkan

    createTransition(state, currentStats(index), MinKanWoAccept, DefaultReward, false)
  }


  def parseAnkan(index: Int, m: Int): Unit = {
    val ankanTile = (m >> 8) & 255

    val origState = currentStats(index).clone()
    acceptTile(currentStats(index), ankanTile)
    val tiles = Array.fill[Int](NumPerTile)(ankanTile)
    updatePlayer(index, tiles, fixTile)
    //    updateBoard(tiles) // Ankan no shown

    createTransition(origState, currentStats(index), AnKanWoAccept, DefaultReward, false)
  }

  // TODO: Don't know difference between kakan and minkan
  def parseKakan(index: Int, m: Int): Unit = {
    val unused = (m >> 5) & 3
    var kakanTile = (m >> 9) & 127

    kakanTile /= 3
    kakanTile *= 4
    kakanTile += unused

    val origState = currentStats(index).clone()

    acceptTile(currentStats(index), kakanTile)
    val tiles = Array.fill[Int](NumPerTile)(kakanTile)
    updatePlayer(index, tiles, fixTile)
    //    updateBoard(tiles.tail)  //TODO: Should ?

    createTransition(origState, currentStats(index), KaKanWoAccept, DefaultReward, false)
  }

  def parsePong(index: Int, m: Int): Unit = {
    var pongTile = (m >> 9) & 127
    val r = pongTile % 3
    pongTile /= 3
    pongTile *= 4
    pongTile += r

    //    logger.info("=====......................... Get pong tile " + pongTile)
    val origState = currentStats(index).clone()
    val tiles = Array.fill[Int](3)(pongTile)
    acceptTile(currentStats(index), pongTile)
    updatePlayer(index, tiles, fixTile)
    updateBoard(tiles.tail)

    createTransition(origState, currentStats(index), PongWoAccept, DefaultReward, false)
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

    val origState = currentStats(index).clone()
    acceptTile(currentStats(index), chowTile)
    updatePlayer(index, candidates, fixTile)
    updateBoard(peers)

    createTransition(origState, currentStats(index), ChowWoAccept, DefaultReward, false)
  }

  def createINDState(state: Array[Int], nextState: Array[Int]): (Array[INDArray], INDArray) = {
    val stateIND = new Array[INDArray](1)
    stateIND(0) = Nd4j.zeros(1, PeerStateLen)
    val nextStateIND = Nd4j.zeros(1, PeerStateLen)

    for(i <- 0 until PeerStateLen) {
      stateIND(0).putScalar(i, state(i))
      nextStateIND.putScalar(i, nextState(i))
    }

    (stateIND, nextStateIND)
  }

  def parseTerminal(node: Node): Unit = {
    //    logger.info("end of game " + gameCount)

    val rewards = node.attribute("sc").map(_.text.split(",").map(_.toInt).zipWithIndex.filter(_._2 % 2 == 1).map(_._1))
      .getOrElse(Array.fill[Int](PlayerNum)(0))
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(-1)
    for (i <- rewards.indices) {
      if (i == who) {
        val machi = node.attribute("machi").map(_.head.text.toInt).getOrElse(-1)
        val state = currentStats(i).clone()
        val fromWho = node.attribute("fromWho").map(_.head.text.toInt).getOrElse(who)
        if (fromWho != who && machi >= 0) {
          acceptTile(currentStats(who), machi)
        }
        createTransition(state, currentStats(i), RonWoAccept, rewards(i), true)
      }else {
        createTransition(currentStats(i), currentStats(i), NOOPWoAccept, rewards(i), true)
      }
    }
  }

  def parseReach(node: Node): Unit = {
    val step = node.attribute("step").map(_.head.text.toInt).getOrElse(0)
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(0)
    step match {
      case 1 =>
        val origState = currentStats(who).clone()

        //TODO: To be fixed
        currentStats.foreach(state => state(ReachOffset + who) = ReachStep1)
        currentStats(who)(ReachOffset) = ReachStep1
        createTransition(origState, currentStats(who), REACHWoAccept, ReachReward, false)
      //        logger.info("---------------------------> Reach " + who)
      case 2 =>
        //TODO: Suppose REACH step = 2 will be sent by server
        //        val origState = currentStats(who).clone()
        currentStats.foreach(state => state(ReachOffset + who) = ReachStep2)
        currentStats(who)(ReachOffset) = ReachStep2
      //Only one reach action executed
      //        createTransition(origState, currentStats(who), REACH, DefaultReward, false)

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
    if (hai >= 0) {
      val doraHai = getDora(hai)
      doraValue(doraHai) = DoraValue
      currentStats.foreach(state => {
        if (state(doraHai) > 0) {
          var tileNum = state(doraHai) & ExtraValueFlag
          tileNum += state(doraHai) / MValue // fixed
          state(doraHai) += tileNum * DoraValue
        }
      })
    }
  }

  def parseDora(node: Node): Unit = {
    val hai = node.attribute("hai").map(_.head.text.toInt / NumPerTile).getOrElse(-1)
    //    val origStates = new Array[Array[Int]](PlayerNum)
    //    for (i <- origStates.indices) {
    //      origStates(i) = currentStats(i).clone()
    //    }

    parseDora(hai)

    //    for(i <- origStates.indices) {
    //      createTransition(origStates(i), currentStats(i), NOOP, DefaultReward, false)
    //    }
  }

  def save(db: ReplayDB): Unit = {
    trans.foreach(tran => db.add(tran))
  }

}
