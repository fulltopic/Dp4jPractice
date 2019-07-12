package dp4jpractice.org.tools.dataprocess.dbprocess.lmdbprocess

import java.io.{ByteArrayInputStream, File}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets

import dp4jpractice.org.tools.dataprocess.dbprocess.caffeproto.Caffe2.{TensorProto, TensorProtos}
import org.lmdbjava.{DbiFlags, Env}

object LmdbTest extends App {
  val DBName = "LmdbTest"
  val DbFilePath = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice"

  def tutorial1(): Unit = {
    val file = new File(DbFilePath)
    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
    val db = env.openDbi(DBName, DbiFlags.MDB_CREATE)

    val key = ByteBuffer.allocateDirect(env.getMaxKeySize)
    val value = ByteBuffer.allocateDirect(100)
    key.put("greeting".getBytes(StandardCharsets.UTF_8)).flip()
    value.put("Hello world".getBytes(StandardCharsets.UTF_8)).flip()
    val valSize = value.remaining()

    db.put(key, value)

    val txn = env.txnRead()
    val found = db.get(txn, key)
    if (found == null) {
      println("Failed to get key " + key)
      return
    }
    val foundValue = txn.`val`()
    println("FoundValue type " + foundValue.getClass)
    if (foundValue.remaining() != valSize) {
      println("No match size " + foundValue.remaining() + " != " + valSize)
      return
    }
    val fetchValue = StandardCharsets.UTF_8.decode(foundValue).toString
    println("FetchValue " + fetchValue)

    db.delete(key)
    val foundAfter = db.get(txn, key)
    if (foundAfter != null) {
      println("Delete failed")
    }

    env.close()
  }

  def buildTensor(): TensorProtos = {
    val tensorProtosBuilder = TensorProtos.newBuilder()

    import scala.collection.JavaConverters._
    val data = List[Integer](1, 2, 3, 4)
    val javaData: java.util.List[Integer] = data.asJava
    val dataProtoBuilder = tensorProtosBuilder.addProtosBuilder()
    dataProtoBuilder.addDims(1).addDims(2).addDims(2).setDataType(TensorProto.DataType.INT32).addAllInt32Data(javaData)
//      .addInt32Data(1).addInt32Data(2).addInt32Data(3).addInt32Data(4)

    val labelProtoBuilder = tensorProtosBuilder.addProtosBuilder()
    labelProtoBuilder.addDims(1).setDataType(TensorProto.DataType.INT32).addInt32Data(3)

    val tensorProtos = tensorProtosBuilder.build()
    println("Get proto count " + tensorProtos.getProtosCount)

//    println("Result " + tensorProtos)

    tensorProtos
  }


  val TensorDbFilePath = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice"
  val TensorDbName = "DbTensor"
  def saveTensor(): Unit = {
    val tensorPtotos = buildTensor()

    val file = new File(TensorDbFilePath)
    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
    val db = env.openDbi(DBName, DbiFlags.MDB_CREATE)

    val key = ByteBuffer.allocateDirect(env.getMaxKeySize).order(ByteOrder.BIG_ENDIAN)
    val value = ByteBuffer.allocateDirect(1000)
    key.put("test".getBytes(StandardCharsets.UTF_8)).flip()
    value.put(tensorPtotos.toByteString.asReadOnlyByteBuffer()).flip()

//    val bytes = tensorPtotos.toByteString
//    for (i <- 0 until 25) {
//      print(bytes.byteAt(i) + ",")
//      if ((i + 1) % 8 == 0) {
//        print("  ")
//      }
//      if ((i + 1) % 32 == 0) {
//        println("")
//      }
//    }
//    println("")


    db.put(key, value)
    db.close()
  }

  val TensorLittleDbName = "DbSmallEndianTensor"
  val TestKey = "test"
  def saveCppTensor(): Unit = {
    val tensorProtos = buildTensor()

    val file = new File(TensorDbFilePath)
    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
    val db = env.openDbi(TensorLittleDbName, DbiFlags.MDB_CREATE)

    val key = ByteBuffer.allocateDirect(env.getMaxKeySize).order(ByteOrder.LITTLE_ENDIAN)
    val value = ByteBuffer.allocateDirect(1000).order(ByteOrder.LITTLE_ENDIAN)
    key.put(TestKey.getBytes(StandardCharsets.UTF_8)).flip()
    value.put(tensorProtos.toByteString.asReadOnlyByteBuffer()).flip()

    db.put(key, value)
    db.close()
  }

  def readCppTensor(): Unit = {
//    val file = new File(DbFilePath)
//    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
//    val db = env.openDbi(TensorLittleDbName, DbiFlags.MDB_CREATE)

    val file = new File("/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/lmdbcpptest")
    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100).setMaxDbs(1).open(file)
    val dbName: String = null
    val db = env.openDbi(dbName, DbiFlags.MDB_CREATE)

    val key = ByteBuffer.allocateDirect(env.getMaxKeySize).order(ByteOrder.LITTLE_ENDIAN)
    key.put("Testtesttesttest".getBytes(StandardCharsets.UTF_8)).flip()
    val txn = env.txnRead()
    val found = db.get(txn, key)
    if (found == null) {
      println("Failed to get key " + key)
      return
    }
    val value = txn.`val`().order(ByteOrder.LITTLE_ENDIAN)
    println("Get value type " + value.asReadOnlyBuffer().getClass)
    println("Get length: " + value.remaining())
    val fetchValue = StandardCharsets.UTF_8.decode(value).toString
    println(fetchValue)
    println(fetchValue.length)

    val bytes = fetchValue.getBytes
    println(bytes.length)
    println(bytes(0) + ", " + bytes(1) + ", " + bytes(2))

    val protos = TensorProtos.parseFrom(bytes)
    println("count " + protos.getProtosCount)

    val dataProto = protos.getProtos(0)
    println("Data proto " + dataProto.getDimsCount)
    for (i <- 0 until dataProto.getDimsCount) {
      print(dataProto.getDims(i) + ", ")
    }
    println("")
    println("Data type " + dataProto.getDataType)
    val dataList = dataProto.getInt32DataList
    for (i <- 0 until dataList.size()) {
      print(dataList.get(i) + ", ")
    }
    println("")

    val labelProto = protos.getProtos(1)
    println("Dim " + labelProto.getDimsCount)
    for (i <- 0 until labelProto.getDimsCount){
      print(labelProto.getDims(i) + ", ")
    }
    println("")
    println("Label " + labelProto.getInt32Data(0))
  }

  def readTensor(): Unit = {
    val file = new File(DbFilePath)
    val env: Env[ByteBuffer] = Env.create().setMapSize(10 * 100 * 100L).setMaxDbs(1).open(file)
    val db = env.openDbi(DBName, DbiFlags.MDB_CREATE)

    val key = ByteBuffer.allocateDirect(env.getMaxKeySize)
    key.put("test".getBytes(StandardCharsets.UTF_8)).flip()
    val txn = env.txnRead()
    val found = db.get(txn, key)
    if (found == null) {
      println("Failed to get key " + key)
      return
    }
    val value = txn.`val`()
    println("Get value type " + value.asReadOnlyBuffer().getClass)
    println("Get length: " + value.remaining())
    val fetchValue = StandardCharsets.UTF_8.decode(value).toString
    println(fetchValue)
    println(fetchValue.length)

    val bytes = fetchValue.getBytes
    println(bytes.length)
    println(bytes(0) + ", " + bytes(1) + ", " + bytes(2))

    val protos = TensorProtos.parseFrom(bytes)
    println("count " + protos.getProtosCount)

    val dataProto = protos.getProtos(0)
    println("Data proto " + dataProto.getDimsCount)
    for (i <- 0 until dataProto.getDimsCount) {
      print(dataProto.getDims(i) + ", ")
    }
    println("")
    println("Data type " + dataProto.getDataType)
    val dataList = dataProto.getInt32DataList
    for (i <- 0 until dataList.size()) {
      print(dataList.get(i) + ", ")
    }
    println("")

    val labelProto = protos.getProtos(1)
    println("Dim " + labelProto.getDimsCount)
    for (i <- 0 until labelProto.getDimsCount){
      print(labelProto.getDims(i) + ", ")
    }
    println("")
    println("Label " + labelProto.getInt32Data(0))
//    val buffer = value.asReadOnlyBuffer()
//    println("Buffer type " + buffer.getClass)
//
//
//    val tensorProtos = TensorProtos.parseFrom(value)
//    println("type " + tensorProtos.getClass)
//    println("tensorProtos: " + tensorProtos)
//    println("proto count " + tensorProtos.getProtosCount)
//    println("has array " + value.hasArray)

//    val array = new Array[Byte](value.remaining() + 4)


//    val input = new ByteArrayInputStream(value.array())
//    val test = TensorProtos.newBuilder().mergeFrom(input)

  }

//  tutorial1()
//  saveTensor()
//  readTensor()
//  saveCppTensor()
  readCppTensor()
}
