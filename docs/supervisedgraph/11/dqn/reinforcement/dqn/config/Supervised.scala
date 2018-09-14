package rl.dqn.reinforcement.dqn.config

object Supervised {
  val PlayerNum: Int = 4
  val NumPerTile = 4
  val TileNum: Int = 34
  val NumPerSort: Int = 9
  val SortNum: Int = 3

  val StateLen: Int = TileNum + 1 + 1 // oya, reach
  val OyaIndex: Int = TileNum
  val ReachIndex: Int = OyaIndex + 1

  val PeerStateLenPerPlayer: Int = TileNum + 2 //reach. oya
  val PeerStateLen: Int = PeerStateLenPerPlayer * (1 + PlayerNum) //mine, on board, reach, myreach, oya
  val ReachOffset: Int = TileNum
  val OyaOffset: Int = ReachOffset + 1
//  val PeerOyaIndex: Int = PeerStateLen - 1
//  val PeerReachIndex: Int = PeerOyaIndex - 1
//  val PeerCommonReach: Int = TileNum * 2



  val ReachReward: Int = -10
  val DefaultReward = 0

  val Accept: Int = TileNum
  val Chow: Int = Accept + 1
  val Pong: Int = Chow + 1
  val KaKan: Int = Pong + 1
  val AnKan: Int = KaKan + 1
  val MinKan: Int = AnKan + 1
  val REACH: Int = MinKan + 1
  val Ron: Int = REACH + 1
  val NOOP: Int = Ron + 1
  val ActionLen: Int = NOOP + 1



  val ChowWoAccept: Int = TileNum  //34
  val PongWoAccept: Int = ChowWoAccept + 1 //35
  val KaKanWoAccept: Int = PongWoAccept + 1 //36
  val AnKanWoAccept: Int = KaKanWoAccept + 1 //37
  val MinKanWoAccept: Int = AnKanWoAccept + 1 //38
  val REACHWoAccept: Int = MinKanWoAccept + 1 //39
  val RonWoAccept: Int = REACHWoAccept + 1 //40
  val NOOPWoAccept: Int = RonWoAccept + 1 //41
  val ActionLenWoAccept: Int = NOOPWoAccept + 1 //42

  val ChowFlag: Int = 1 << 2
  val PongFlag: Int = 1 << 3
  val KakanFlag: Int = 1 << 4
  val KitaBits: Int = 5
  val AnkanFlag = 3

  val DoraValue = 8
  val MValue = 64 //fixed
  val ExtraValueFlag = 7

  val DoraInSeed = 5

  val ReachStep1 = 1
  val ReachStep2 = 2

  val AkaValues = Map(16 -> DoraValue, 52 -> DoraValue, 88 -> DoraValue)
}
