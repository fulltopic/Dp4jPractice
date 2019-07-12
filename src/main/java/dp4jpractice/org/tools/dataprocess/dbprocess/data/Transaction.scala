package dp4jpractice.org.tools.dataprocess.dbprocess.data

class Transaction(currStates: Array[MjPlayerState], currAction: Int) {
  val states = new Array[MjPlayerState](DbConsts.StatePlayerNum)
  for (i <- states.indices) {
    states(i) = new MjPlayerState()
  }
  val action = currAction

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

//    val n: Int = DbConsts.StateLen / 10
//    for (i <- 0 until n) {
//      for (player <- 0 until DbTenhouConsts.PlayerNum) {
//        for (j <- 0 until 10) {
//          buffer.append(states(player).get(j + i * 10))
//          buffer.append(",")
//        }
//        buffer.append("\n")
//      }
//      buffer.append("\n")
//    }
//
//
//    for (player <- 0 until DbTenhouConsts.PlayerNum) {
//      for (i <- (n * 10) until DbConsts.StateLen) {
//        buffer.append(states(player).get(i)).append(",")
//      }
//      buffer.append("\n")
//    }

//    for (i <- 0 until 6) {
//      for (player <- 0 until DbTenhouConsts.PlayerNum) {
//        for (tile <- (9 * i) until (9 * (i + 1))) {
//          buffer.append(states(player).get(tile))
//          buffer.append(",")
//        }
//        buffer.append("\n")
//      }
//      buffer.append("\n")
//    }
//
//    for (player <- 0 until DbTenhouConsts.PlayerNum) {
//      for (tile <- 54 until DbConsts.StateLen) {
//        buffer.append(states(player).get(tile))
//        buffer.append(",")
//      }
//      buffer.append("\n")
//    }
//    buffer.append("\n")
//
//    for (player <- 0 until DbTenhouConsts.PlayerNum) {
//      for (tile <- 54 until DbConsts.StateLen) {
//        buffer.append(states(player).get(tile))
//        buffer.append(",")
//      }
//      buffer.append("\n")
//    }
    buffer.append("\n")

    buffer.toString()
  }
}

object Transaction {
  def getDims(): List[Int] = List[Int](1, DbConsts.StatePlayerNum, DbConsts.StateLen)
}