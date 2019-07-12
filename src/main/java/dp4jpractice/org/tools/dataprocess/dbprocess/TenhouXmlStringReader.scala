package dp4jpractice.org.tools.dataprocess.dbprocess

import com.sun.media.jfxmedia.events.PlayerStateEvent.PlayerState
import akka.event.slf4j.Logger
import scala.xml.{Node, Text, XML}
import dp4jpractice.org.tools.dataprocess.dbprocess.data._

class TenhouXmlStringReader(rawBuffer: String) {
  val logger = Logger("TenhouXmlStringReader")

  val currState = new Array[MjPlayerState](DbConsts.StatePlayerNum)
  for (i <- currState.indices) {
    currState(i) = new MjPlayerState()
  }
  var currScene = new SceneRecord()
  var scenes = List.empty[SceneRecord]

  var viewer: Int = 0
  var gameCount: Int = 0
//  val doraValue = Array.fill[Int](PeerStateLen)(0)



  def initCurrScene(): Unit = {
    //TODO: Make sure currenScene is a new object
    currScene = new SceneRecord()
  }

  def raw2Tile(tileNum: Int): Int = tileNum / DbTenhouConsts.NumPerTile
  def getPlayerIndex(player: Int): Int = (player - viewer + DbTenhouConsts.PlayerNum) % DbTenhouConsts.PlayerNum + 1

  def readFile(): List[SceneRecord] = {
    for (player <- 0 until DbTenhouConsts.PlayerNum) {
      viewer = player
      logger.debug("-------------------------> To read file as player " + viewer)

      val root = XML.loadString(rawBuffer)
//      val root = XML.loadFile("/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/datasets/mjsupervised/xmlfiles/smalltest/")
//      val root = XML.loadFile("/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/datasets/mjsupervised/xmlfiles/smalltest/mj/test.xml")
      val a = root \ "Scene"
//      logger.debug("" + root)
//      logger.debug("Get game size " + a.length)
      a.foreach(readScene)

      val b = root \ "SCENE"
//      logger.debug("Get game size " + b.length)
      b.foreach(readScene)
    }

    scenes
  }


  def readScene(sceneNode: Node): Unit = {
    gameCount += 1

      logger.debug("Read game " + gameCount + " as plyer " + viewer)

      val byeNode = sceneNode \ "BYE"
      if (byeNode.length > 0) {
        //      logger.debug("--------------------------> Interrupted game, give up")
      } else {
        sceneNode.child.foreach { child => {
          child match {
            case _: Text => //logger.debug(child.head.label)
            case _: Node => readNodeInSeq(child)
          }
        }
        }
      }
  }

  def readNodeInSeq(node: Node): Unit = {
    //    logger.debug("===========================> Get a node " + node.label)
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

  def parseDora(node: Node): Unit = {
    logger.debug("Dora, unsupported " + node.label)
  }

  def parseUN(node: Node): Unit = {
    logger.debug("Don't know what to do with UN")
  }

  def parseInit(node: Node): Unit = {
//    logger.debug("Get currState size " + currState.length)

    for (i <- currState.indices) {
//      logger.debug("Reset state " + i)
      currState(i).reset()
    }

    val ten = node.attribute("ten")

    // oya
    val oya = node.attribute("oya").map(_.head.text.toInt).getOrElse(0)

    if (oya == viewer) {
      currState(0).addTile(DbConsts.OyaPos, 1)
    } else {
      currState(getPlayerIndex(oya)).addTile(DbConsts.OyaPos, 1)
    }
    //    logger.debug(oya)

    // dora
//    val dora = node.attribute("seed").map(_.head.text.split(",").map(_.toInt).apply(DbTenhouConsts.DoraInSeed)).getOrElse(-1)
//    parseDora(dora / DbTenhouConsts.NumPerTile)

    val haiName = "hai" + viewer.toString
    val hai = node.attribute(haiName)
    parseInitPlayer(hai, viewer)
  }

  def parseInitPlayer(hai: Option[Seq[Node]], index: Int): Unit = {
    hai match {
      case Some(x) => {
//        logger.info("Init tile " + x)
        x.head.text.split(",").map(_.toInt).foreach(rawTile => {
          currState(0).addTile(raw2Tile(rawTile), 1)
        })
//        logger.debug("After init " + currState(0))
      }
      case _ => //nothing
    }
  }

  val flag2Player = Map[Char, Int]('T' -> 0, 'D' -> 0, 'U' -> 1, 'E' -> 1, 'V' -> 2, 'F' -> 2, 'W' -> 3, 'G' -> 3)
  val acceptFlags = Set('T', 'U', 'V', 'W')
  val dropFlags = Set('D', 'E', 'F', 'G')

  def parsePlayer(node: Node): Unit = {
    logger.debug("player " + node.label)
    val flag = node.label.charAt(0)
    val origTile = node.label.substring(1).toInt

    if (acceptFlags.contains(flag) && flag2Player.getOrElse(flag, -1) == viewer) {
      currState(0).addTile(raw2Tile(origTile), 1)
    } else if (dropFlags.contains(flag)) {
      val playerIndex = flag2Player.getOrElse(flag, -1)
      val tile = raw2Tile(origTile)
      if (flag2Player.getOrElse(flag, -1) == viewer) {
        createTransaction(tile)
        currState(0).rmTile(tile, 1)
        currState(1).addTile(tile, 1)
      } else {
        currState(getPlayerIndex(playerIndex)).addTile(tile, 1)
      }
    }
  }


  protected def createTransaction(action: Int): Unit = {
//    logger.debug("State before trans \n")
//    logger.debug(currState(0).toString)
    val trans = new Transaction(currState, action)
//    logger.debug("The new trans \n")
//    logger.debug(trans.toString)
    currScene.addTrans(trans)
  }


  //TODO: Parse steal should create transition
  def parseSteal(node: Node): Unit = {
    //    logger.debug("steal " + node.text)
    val whoAttr = node.attribute("who")
    whoAttr match {
      case Some(x) => {
        val who = x.head.text.toInt
        val mAttr = node.attribute("m")
        mAttr match {
          case Some(y) => {
            val m = y.text.toInt

            if ((m & DbTenhouConsts.ChowFlag) > 0) {
              parseChow(who, m)
            }else if ((m & DbTenhouConsts.PongFlag) > 0) {
              parsePong(who, m)
            }else if ((m & DbTenhouConsts.KakanFlag) > 0) {
              parseKakan(who, m)
            }else if ((m & DbTenhouConsts.AnkanFlag) == 0) {
              parseAnkan(who, m)
            }else if ((m >> DbTenhouConsts.KitaBits & 1) == 1) {
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

  // mjlog_pf4-20_n2/2010033123gm-0029-0000-53c6c5e8&tw=0.xml
  def parseKita(index: Int, m: Int): Unit = {
    //To throw exception
    logger.debug("Want to process kita: " + m)
    logger.debug("----------------------------------------> Kita has not been implemented " + currState(DbConsts.StatePlayerNum - 1).get(0))
  }

  def parseMinkan(index: Int, m: Int): Unit = {
    //exception would be thrown
    val kanTile = (m >> 8) & 255
    logger.debug("Get kanTile as " + kanTile)
    val player = getPlayerIndex(index)
    if (index == viewer) {
      createTransaction(DbConsts.MinkanAction)
      currState(0).rmTile(raw2Tile(kanTile), 3)
      currState(0).addTile(raw2Tile(kanTile) + DbTenhouConsts.TileNum, 4)
    } else {
      currState(player).addTile(raw2Tile(kanTile) + DbTenhouConsts.TileNum, 4)
    }
  }


  //TODO: Check if after an accept
  def parseAnkan(index: Int, m: Int): Unit = {
    val ankanTile = (m >> 8) & 255
    logger.debug("Get ankanTile " + ankanTile)
    val player = getPlayerIndex(index)

    if (index == viewer) {
      createTransaction(DbTenhouConsts.AnkanFlag)
      currState(0).rmTile(raw2Tile(ankanTile), 4)
      currState(0).addTile(raw2Tile(ankanTile) + DbTenhouConsts.TileNum, 4)
    } else {
      currState(player).addTile(raw2Tile(ankanTile) + DbTenhouConsts.TileNum, 4)
    }
  }

  // TODO: Don't know difference between kakan and minkan
  def parseKakan(index: Int, m: Int): Unit = {
    val unused = (m >> 5) & 3
    var kakanTile = (m >> 9) & 127

    kakanTile /= 3
    kakanTile *= 4
    kakanTile += unused
    logger.debug("Get kankan tile " + kakanTile)
    val player = getPlayerIndex(index)

    if (index == viewer) {
      createTransaction(DbConsts.KakanAction)
      currState(0).rmTile(raw2Tile(kakanTile), 3)
      currState(0).addTile(raw2Tile(kakanTile) + DbTenhouConsts.TileNum, 4)
    } else {
      currState(player).addTile(raw2Tile(kakanTile) + DbTenhouConsts.TileNum, 4)
    }
  }

  //TODO: The special value for action indication
  def parsePong(index: Int, m: Int): Unit = {
    var pongTile = (m >> 9) & 127
    val r = pongTile % 3
    pongTile /= 3
    pongTile *= 4
    pongTile += r
    logger.debug("Pong tile before " + pongTile)

    pongTile = raw2Tile(pongTile)
    logger.debug("Get pong tile " + pongTile)
    val player = getPlayerIndex(index)

    if (index == viewer) {
      createTransaction(DbConsts.PongAction)
      currState(0).rmTile(pongTile, 2)
      currState(0).addTile(pongTile + DbTenhouConsts.TileNum, 3)
    } else {
      currState(player).addTile(pongTile + DbTenhouConsts.TileNum, 3)
    }
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
    logger.debug("Get chow candidates " + chowTile + ", " + peers(0) + ", " + peers(1))

    val player = getPlayerIndex(index)
    if (index == viewer) {
      createTransaction(DbConsts.ChowAction)
      peers.foreach(p => currState(0).rmTile(raw2Tile(p), 1))
      peers.foreach(p => currState(0).addTile(raw2Tile(p) + DbTenhouConsts.TileNum, 1))
      currState(0).addTile(raw2Tile(chowTile) + DbTenhouConsts.TileNum, 1)
    } else {
      peers.foreach(p => currState(player).addTile(raw2Tile(p) + DbTenhouConsts.TileNum, 1))
      currState(player).addTile(raw2Tile(chowTile) + DbTenhouConsts.TileNum, 1)
    }
  }

  //TODO: create transition
  def parseTerminal(node: Node): Unit = {
    logger.debug("end of game " + gameCount)

    val rewards = node.attribute("sc").map(_.text.split(",").map(_.toInt).zipWithIndex.filter(_._2 % 2 == 1).map(_._1))
      .getOrElse(Array.fill[Int](DbTenhouConsts.PlayerNum)(0))
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(-1)
    if (who == viewer) {
      createTransaction(DbConsts.RonAction)
      //TODO: To update states after termination?
    } else {
      createTransaction(DbConsts.NoopAction)
    }
    scenes = scenes :+ currScene
    initCurrScene()

//    for (i <- rewards.indices) {
//      if (i == who) {
//        val machi = node.attribute("machi").map(_.head.text.toInt).getOrElse(-1)
//        val state = currentStats(i).clone()
//        val fromWho = node.attribute("fromWho").map(_.head.text.toInt).getOrElse(who)
//        if (fromWho != who && machi >= 0) {
//          acceptTile(currentStats(who), machi)
//        }
//
//        createDataPair(RonWoAccept, i)
//      }else {
//        createDataPair(NOOPWoAccept, i)
//      }
//    }
  }

  //TODO: Create transition
  def parseReach(node: Node): Unit = {
    val step = node.attribute("step").map(_.head.text.toInt).getOrElse(0)
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(0)
    step match {
      case 1 =>
        if (who == viewer) {
          createTransaction(DbConsts.ReachAction)
          currState(0).addTile(DbConsts.ReachPos, 1)
        } else {
          currState(getPlayerIndex(who)).addTile(DbConsts.ReachPos, 1)
        }
      case 2 =>
        logger.debug("Reach step2")
//        currentStats.foreach(state => state(PeerCommonReach + who) = ReachStep2)
//        currentStats(who)(PeerReachIndex) = ReachStep2

      case _ => logger.debug("Received invalid reach step " + step)
    }
  }


//  def getDora(hai: Int): Int = {
//    hai match {
//      case tile if tile >= 0 && tile < 27 =>
//        val tmp = tile / 9
//        (tile + 1) % 9 + tmp * 9
//      case tile if tile >= 27 && tile < 31 =>
//        val tmp = tile - 27
//        (tmp + 1) % 4 + 27
//      case _ =>
//        val tmp = hai - 31
//        (hai + 1) % 3 + 31
//    }
//  }
//
//  def parseDora(hai: Int): Unit = {
//    if (hai >= 0) {
//      val doraHai = getDora(hai)
//      doraValue(doraHai) = DoraValue
//      currState.foreach(state => {
//        if (state(doraHai) > 0) {
//          var tileNum = state(doraHai).toInt & DbTenhouConsts.ExtraValueFlag
//          tileNum += state(doraHai).toInt / MValue // fixed
//          state(doraHai) += tileNum * DoraValue
//        }
//      })
//    }
//  }
//
//  def parseDora(node: Node): Unit = {
//    val hai = node.attribute("hai").map(_.head.text.toInt / DbTenhouConsts.NumPerTile).getOrElse(-1)
//    parseDora(hai)
//  }
}