package dp4jpractice.org.tools.dataprocess.dbprocess.data

class MjPlayerState {
  val state = Array.fill[Int](DbConsts.StateLen)(0)

  def addTile (tileIndex: Int, tileNum: Int): Unit = {
    state(tileIndex) += tileNum //TODO: Check it
  }

  def rmTile (tileIndex: Int, tileNum: Int): Unit = {
    state(tileIndex) -= tileNum
  }

  def setReach(): Unit = {
    state(DbConsts.ReachPos) = 1
  }

  def setOya(): Unit = {
    state(DbConsts.OyaPos) = 1
  }

  def setWinner(): Unit = {
    state(DbConsts.WinPos) = 1
  }

  def reset(): Unit = {
    for (i <- state.indices) {
      state(i) = 0
    }
  }

  def get(index: Int): Int = state(index)
  def set(index: Int, tileNum: Int): Unit = {
    state(index) = tileNum
  }

  override def toString: String = {
    val buffer = new StringBuilder
    for (i <- 0 until state.length) {
      if ((i + 1) % 8 == 0) {
        buffer.append("\n")
      }
      buffer.append(state(i)).append(",")
    }

    buffer.toString()
  }
}