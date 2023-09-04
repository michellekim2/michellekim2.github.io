---
layout: page
title: CIS 053 Intro to Machine Learning Final Project
description: Use Python, Scikit to perform feature selection, lasso and ridge regression, cross validation on housing price data
img: assets/img/Final Project CIS 053 (Michelle Kim).jpg
importance: 1
category: work
---

Final project to culminate all content mastered over the summer term. 

Programming Tasks: descriptive statistic and generate plots including correlation heatmap, perform manual analysis of plots for potential relevant features, perform feature selection with Recursive Feature Elimination, build regularized regression model (Lasso and Ridge methods), use K-fold method for cross validation

Produced Final Report concisely presenting data and process.

<iframe src="assets/pdf/CIS053-Written-Final.pdf" width="100%" height="500px"></iframe>

<iframe src="assets/pdf/example_pdf.pdf" width="100%" height="500px"></iframe>

<script>
  // Initialize PDF.js
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.582/pdf.worker.min.js';

  // Load and render the PDF
  var pdfViewer = document.getElementById('pdfViewer');
  var pdfFile = 'assets/pdf/CIS053-Written-Final.pdf'; // Replace with the actual path to your PDF file

  var loadingTask = pdfjsLib.getDocument(pdfFile);
  loadingTask.promise.then(function (pdfDocument) {
    // Initialize the PDF viewer
    var pdfViewer = new pdfjsViewer.PDFViewer({
      container: pdfViewer,
    });
    pdfViewer.setDocument(pdfDocument);
  });
</script>



Here is a link to the full code (Juypter Notebook):
<a href="https://github.com/michellekim2/portfolio/blob/main/CIS053-Final-Code.ipynb">Link to full code</a>


