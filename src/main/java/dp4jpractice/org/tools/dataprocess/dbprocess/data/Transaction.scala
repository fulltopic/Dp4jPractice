package dp4jpractice.org.tools.dataprocess.dbprocess.data

class Transaction(currStates: Array[MjPlayerState], currAction: Int) {
  val states = new Array[MjPlayerState](DbConsts.StatePlayerNum)
  for (i <- states.indices) {
    states(i) = new MjPlayerState()
  }
  val action: Int = currAction

  //TODO: Scala clone
  for (i <- states.indices) {
    for (j <- 0 until DbConsts.StateLen) {
      states(i).set(j, currStates(i).get(j))
    }
  }

  override def toString: String = {
    val buffer = new StringBuilder
    buffer.append("Action = ").append(action).append("\n")

    for (player <- states.indices) {
      for (i <- 0 until DbTenhouConsts.TileNum) {
        buffer.append(states(player).get(i)).append(",")
        if ((i + 1) % 10 == 0) {
          buffer.append("  ")
        }
      }
      buffer.append("\n")
    }
    buffer.append("\n")

    for (player <- states.indices) {
      for (i <- DbTenhouConsts.TileNum until (2 * DbTenhouConsts.TileNum)) {
        buffer.append(states(player).get(i)).append(",")
        if ((i + 1 - DbTenhouConsts.TileNum) % 10 == 0) {
          buffer.append("  ")
        }
      }
      buffer.append("\n")
    }
    buffer.append("\n")

    for (player <- states.indices) {
      for (i <- (2 * DbTenhouConsts.TileNum) until DbConsts.StateLen) {
        buffer.append(states(player).get(i)).append(",")
      }
      buffer.append("\n")
    }

    buffer.append("\n")

    buffer.append("\n")

    buffer.toString()
  }
}

object Transaction {
  def getDims(): List[Int] = List[Int](1, DbConsts.StatePlayerNum, DbConsts.StateLen)
}