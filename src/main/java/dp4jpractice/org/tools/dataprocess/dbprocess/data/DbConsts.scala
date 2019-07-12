package dp4jpractice.org.tools.dataprocess.dbprocess.data

object DbConsts {
  val StatePlayerNum: Int = DbTenhouConsts.PlayerNum + 1

  val ReachPos: Int = DbTenhouConsts.TileNum * 2
  val OyaPos: Int = ReachPos + 1
  val WinPos: Int = OyaPos + 1
  val StateLen: Int = WinPos + 1

  val ChowAction: Int = DbTenhouConsts.TileNum
  val PongAction: Int = ChowAction + 1
  val KakanAction: Int = PongAction + 1
  val MinkanAction: Int = KakanAction + 1
  val AnkanAction: Int = MinkanAction + 1
  val ReachAction: Int = AnkanAction + 1
  val RonAction: Int = ReachAction + 1
  val NoopAction: Int = RonAction + 1
  val ActionLen: Int = NoopAction + 1
}