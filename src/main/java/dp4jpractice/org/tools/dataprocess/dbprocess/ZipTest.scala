package dp4jpractice.org.tools.dataprocess.dbprocess

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream, FileOutputStream, IOException, PrintWriter}
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.zip.{GZIPInputStream, ZipEntry, ZipFile, ZipInputStream, ZipOutputStream}

import dp4jpractice.org.tools.dataprocess.dbprocess.lmdbprocess.LmdbOperator
import sun.misc.IOUtils

import scala.io.Source

object ZipTest extends App {
  val zipFilePath: String = "/home/zf/workspaces/workspace_cpp/testcaffe2/res/mjdb/zips/mjlog_pf4-20_n1.zip"


  def testExtract(): Unit = {
    val files = ExtractFiles.unzip(zipFilePath)
    //  files.map(entry => {
    //    println(entry._1)
    //    println(entry._2.length)
    //  })
    //  val entry = files.head
    //  println("File " + entry._1)
    //  val output = new PrintWriter(new File("./testzipoutput.xml"))
    //  output.write(genXml(entry._2))
    //  output.close()
    //  println(entry._2)
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
    val files = ExtractFiles.unzip(zipFilePath)
    println("Read file of " + files.head._1)

    val reader = new TenhouXmlStringReader(files.head._1, files.head._2)
    val scenes = reader.readFile()

//    println(scenes.head)
  }

  def testReadZipFiles(): Unit = {
    val files = ExtractFiles.unzip(zipFilePath)

    for ((k, v) <- files) {
      println("Read file " + k)
      val reader = new TenhouXmlStringReader(k, v)
      val scenes = reader.readFile()
      println("End of file " + k + " get scenes " + scenes.length)
    }
  }

  def testSaveDb(): Unit = {
    val files = ExtractFiles.unzip(zipFilePath)
    println("Read file of " + files.head._1)

    val reader = new TenhouXmlStringReader(files.head._1, files.head._2)
    val scenes = reader.readFile()

    val TensorDbFilePath = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/lmdbscenetest/"
//    val DbName = "TensorDb"
    val DbName: String = null
    val dbOp = new LmdbOperator(TensorDbFilePath, DbName, ByteOrder.BIG_ENDIAN)
    dbOp.saveTensor(scenes.head)
    dbOp.close()
  }

  def testReadDb(): Unit = {
     val TensorDbFilePath = "/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice"
    val DbName = "TensorDb"
    val dbOp = new LmdbOperator(TensorDbFilePath, DbName, ByteOrder.LITTLE_ENDIAN)
    dbOp.readFirstOne()
    dbOp.close()
  }

//  testExtract()
//  testReadFile()
//  testReadZipFiles()
  testSaveDb()
//  testReadDb()
}