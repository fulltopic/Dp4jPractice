package dp4jpractice.org.tools.dataprocess.dbprocess

import java.nio.ByteOrder

import dp4jpractice.org.tools.dataprocess.dbprocess.lmdbprocess.LmdbOperator

import scala.io.Source

object ZipTest extends App {
  val ZipFilePath: String = "/home/zf/workspaces/res/mjzips/zips/mjlog_pf4-20_n1.zip"
  val TensorDbFilePath = "/home/zf/workspaces/res/dbs/lmdbscenetest"


  def testExtract(): Unit = {
    val files = ExtractFiles.unzip(ZipFilePath)
      files.foreach(entry => {
        println(entry._1)
        println(entry._2.length)
      })
      val entry = files.head
      println("File " + entry._1)
      println(entry._2)
  }

  def testReadFile(): Unit = {
    val fileName: String = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/test.xml"
    val content = Source.fromFile(fileName).mkString

    println("------------------------> Content file " + content.length)
    val reader = new TenhouXmlStringReader(fileName, content)
    val scenes = reader.readFile()
    println(scenes.length)


    scenes.foreach(scene => println(scene))
  }

  def testReadZipFile(): Unit = {
    val files = ExtractFiles.unzip(ZipFilePath)
    println("Read file of " + files.head._1)

    val reader = new TenhouXmlStringReader(files.head._1, files.head._2)
    val scenes = reader.readFile()

    println(scenes.head)
  }

  def testReadZipFiles(): Unit = {
    val files = ExtractFiles.unzip(ZipFilePath)

    for ((k, v) <- files) {
      println("Read file " + k)
      val reader = new TenhouXmlStringReader(k, v)
      val scenes = reader.readFile()
      println("End of file " + k + " get scenes " + scenes.length)
    }
  }

  def testSaveDb(): Unit = {
    val files = ExtractFiles.unzip(ZipFilePath)
    println("Read file of " + files.head._1)

    val reader = new TenhouXmlStringReader(files.head._1, files.head._2)
    val scenes = reader.readFile()

//    val DbName = "TensorDb"
    val DbName: String = null
    val dbOp = new LmdbOperator(TensorDbFilePath, DbName, ByteOrder.BIG_ENDIAN)

    scenes.foreach(scene => dbOp.saveTensor(scene))
//    dbOp.saveTensor(scenes.head)
//    dbOp.saveTensor(scenes.tail.head)
    dbOp.close()
  }

  def saveDb(): Unit = {
    val DbName: String = null
    val dbOp = new LmdbOperator(TensorDbFilePath, DbName, ByteOrder.BIG_ENDIAN)
    val Num = 1000
    var num: Long = 0

    val files = ExtractFiles.unzip(ZipFilePath);

    files.foreach(file => {
      val reader = new TenhouXmlStringReader(file._1, file._2)
      val scenes = reader.readFile()

      scenes.foreach(scene => dbOp.saveTensor(scene))
      num += scenes.size
      println("----------------------------------------------------------_> Num: " + num)
    })
//    for (file <- files) {
//      if (num < Num) {
//        val reader = new TenhouXmlStringReader(file._1, file._2)
//        val scenes = reader.readFile()
//
//        num += scenes.size
//        scenes.foreach(scene => dbOp.saveTensor(scene))
//      }
//    }
    dbOp.close()
  }

  def testReadDb(): Unit = {
//     val TensorDbFilePath = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice"
    val DbName = "TensorDb"
    val nullDbName: String = null
    val dbOp = new LmdbOperator(TensorDbFilePath, nullDbName, ByteOrder.LITTLE_ENDIAN)
    dbOp.readFirstOne()
    dbOp.close()
  }

//  testExtract()
//  testReadFile()
//  testReadZipFiles()
//  println(scala.runtime.toString)
//  testSaveDb()
//  testReadDb()

  saveDb()
}