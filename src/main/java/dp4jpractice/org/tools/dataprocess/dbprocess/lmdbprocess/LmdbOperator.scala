package dp4jpractice.org.tools.dataprocess.dbprocess.lmdbprocess

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.JavaConverters._
import dp4jpractice.org.tools.dataprocess.dbprocess.caffeproto.Caffe2.{TensorProto, TensorProtos}
import dp4jpractice.org.tools.dataprocess.dbprocess.data.{DbConsts, DbTenhouConsts, SceneRecord, Transaction}
import org.lmdbjava.{DbiFlags, Env, SeekOp}

class LmdbOperator(parentPath: String, dbName: String, order: ByteOrder) {
  val file = new File(parentPath)
  val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
  val db = env.openDbi(dbName, DbiFlags.MDB_CREATE)
  val javaDims: java.util.List[java.lang.Long] = Transaction.getDims().map(d => new java.lang.Long(d)).asJava

  def buildTensor(tran: Transaction): TensorProtos = {
    val tensorProtosBuilder = TensorProtos.newBuilder()

    val dataProtoBuilder = tensorProtosBuilder.addProtosBuilder()
    dataProtoBuilder.addAllDims(javaDims)
//    dataProtoBuilder.addDims(1).addDims(DbConsts.StatePlayerNum).addDims(DbConsts.StateLen)
    dataProtoBuilder.setDataType(TensorProto.DataType.INT32)
    val scalaData = tran.states.flatMap(_.state).toList.map(d => new Integer(d))
    val javaData: java.util.List[Integer] = scalaData.asJava
    dataProtoBuilder.addAllInt32Data(javaData)

    val labelProtoBuilder = tensorProtosBuilder.addProtosBuilder()
    labelProtoBuilder.addDims(1).setDataType(TensorProto.DataType.INT32).addInt32Data(tran.action)
    println("Put label " + tran.action)

    tensorProtosBuilder.build()
  }

  def saveTensor(dbKey: String, scene: SceneRecord): Unit = {
    var step: Int = 0
    val key = ByteBuffer.allocateDirect(env.getMaxKeySize).order(order)
    val value = ByteBuffer.allocateDirect(1000).order(order)

    val trans = scene.getTrans()

    for (tran <- trans) {
      val tensor = buildTensor(tran)


      key.put((dbKey + "_" + step.toString).getBytes(StandardCharsets.UTF_8)).flip()
      value.put(tensor.toByteString.asReadOnlyByteBuffer()).flip()

      db.put(key, value)

      println("Put tran for step " + step)
      step += 1
    }
  }

  def readFirstOne(): Unit = {
    val key = ByteBuffer.allocateDirect(env.getMaxKeySize).order(order)
    val value = ByteBuffer.allocateDirect(1000).order(order)

    val txn = env.txnRead()
    val cursor = db.openCursor(txn)

    cursor.seek(SeekOp.MDB_FIRST)
    println("Seek key " + StandardCharsets.UTF_8.decode(cursor.key()).toString)
    db.close()
  }

  def close(): Unit = db.close()
}
