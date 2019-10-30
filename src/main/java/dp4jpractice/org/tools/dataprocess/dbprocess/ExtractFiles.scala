package dp4jpractice.org.tools.dataprocess.dbprocess

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream, FileOutputStream, IOException, PrintWriter}
import java.nio.charset.StandardCharsets
import java.util.zip.{GZIPInputStream, ZipEntry, ZipFile, ZipInputStream, ZipOutputStream}

//import sun.misc.IOUtils

import scala.io.Source

object ExtractFiles {
  def parseZip(zis: ZipInputStream, files: scala.collection.mutable.Map[String, String]): Unit = {
    val ze = zis.getNextEntry
    if (ze != null) {
      val fileName = ze.getName
      val gisBuffer = new ByteArrayOutputStream()
      val buffer = new Array[Byte](1024)

      var len = zis.read(buffer)
      while (len > 0) {
        gisBuffer.write(buffer, 0, len)
        len = zis.read(buffer)
      }

      val gis = new GZIPInputStream(new ByteArrayInputStream(gisBuffer.toByteArray))
      val outputBuffer = new ByteArrayOutputStream()

      len = gis.read(buffer)
      while (len > 0) {
        outputBuffer.write(buffer, 0, len)
        len = gis.read(buffer)
      }


      files += (fileName -> genXml(outputBuffer.toString))
      parseZip(zis, files)
    }
  }

  def unzip(zipFileName: String): scala.collection.mutable.Map[String, String] = {
    val zis: ZipInputStream = new ZipInputStream(new FileInputStream(zipFileName))
    var files = scala.collection.mutable.Map[String, String]()

    parseZip(zis, files)

    files
  }


  def genXml(content: String): String = {
    val sceneContent = content.replaceAll("<INIT","\n</SCENE>\n<SCENE>\n<INIT")
    val logContent = sceneContent.replaceAll("</mjloggm>", "\n</SCENE>\n</mjloggm>")
    val rmContent = logContent.replaceFirst("</SCENE>", "")

    rmContent
  }
}