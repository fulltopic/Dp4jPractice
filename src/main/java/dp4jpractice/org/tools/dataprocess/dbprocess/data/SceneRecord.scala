package dp4jpractice.org.tools.dataprocess.dbprocess.data

//TODO: Mutable to immutable
class SceneRecord {
  protected var ts = List.empty[Transaction]

  def addTrans(trans: Transaction): Unit = {
    ts = ts :+ trans
  }

  def getTrans(): List[Transaction] = ts

  override def toString: String = {
    val buff = new StringBuilder
    var i: Int = 0
    buff.append("---------------------------> Scene --------------------------> \n")
    for (trans <- ts) {
      buff.append("Trans ").append(i).append("\n")
      buff.append(trans.toString)
      i += 1
    }

    buff.append("End of trans \n")

    buff.toString()
  }
}
