# Configuration for report directory - uses bibliography
$aux_dir = "out";
$out_dir = "out";
ensure_path($aux_dir);

# Enable bibliography processing
$bibtex_use = 2;
$pdf_mode = 1;

# Copy PDF back to main directory  
$success_cmd = 'cp out/main.pdf .';

# Clean up auxiliary files but keep PDF
$cleanup_mode = 1;