package rl.dqn.reinforcement.dqn

import java.util

import akka.event.slf4j.Logger
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.Random

object MessageParseUtils{
  private val logger = Logger("MessageParseUtils")
  

  val ActionStep0: Int = 0
  val ActionStep1: Int = 1
  val ActionStep2: Int = 2
  val InvalidAction: Int = -1
  val InvalidTile: Int = -1


  val jsonActionKey = "actions"
  val jsonTileKey = "tile"
  private[this] val keys = Set[String]("TAIKYOKU", " INIT", "T", "D", "U", "E", "V", "F", "W", "G", "N", "RYUUKYOKU", "AGARI", "REACH", "DORA")
  private[this] val sceneKeys = Set[String]("HELO", "JOIN", "REJOIN", "UN", "LN", "GO", "PROF")
  private[this] val actionTitle = Map[Int, (String, String)](0 -> ("T", "D"), 1 -> ("U", "E"), 2 -> ("V", "F"), 3 -> ("W", "G"))
  private[this] val gameMsgSet = scala.collection.immutable.Set[String]("INIT", "DORA", "REACH", "AGARI", "N", "RYUUKYOKU", "REINIT", "SAIKAI", "TAIKYOKU") //, "PROF"
  val dropKeyPattern = "[e, f, g, E, F, G][0-9]+"
  val acceptKeyPattern = "[u, v, w, U,V,W]"
  val myAcceptPattern = "[T, t][0-9]+"
  val myDropPattern = "[D][0-9]+"
  val tileKeyPattern = "[d, u, e, v, f, w, t, g, D,U,E,V,F,W,T,G][0-9]+" //efgd for drop, uvwt for accept
  val ronActionFlags = Set[Int](8, 9, 10, 11, 12, 13, 15, 16)

  val kanValue: Double = 0.1
  val chowValue: Double = 0.25
  val pongValue: Double = 0.5
  val reachValue: Double = 0.5

  val StartConnection = "StartConnection"
  val StartConnectionRsp = "SentStartConnection"
  val CloseConnection = "CloseConnection"
  val ClosedConnection = "ClosedConnection"
  val SendGameEndReply = "SendGameEndReply"
//  val AbortedConnection = "AbortedConnection"
  val ResetAction: Int = -1
  val LNLIMIT: Int = 10
  val ReinitReach: Int = 255

  val actionRandom: Random = new Random(177)

}
