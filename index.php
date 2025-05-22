<?php
require('../../config.php');
require_login();
$PAGE->set_url(new moodle_url('/local/chatbot/index.php'));
$PAGE->set_context(context_system::instance());
$PAGE->set_title("RAG Chatbot");
$PAGE->set_heading("AI Destekli Chatbot");

echo $OUTPUT->header();

$input = optional_param('q', '', PARAM_TEXT);

if ($input !== '') {
    putenv("GOOGLE_API_KEY=AIzaSyAyHrG566VKpTroXOIm3aYumeXQ8mS2KOQ");
    
    $python = '/home/yavuzsssvr/chatbotenv/bin/python';
    // localimde oluşturduğum chatbot python scriptini kullanıyor
    $script = '/var/www/html/moodle/local/chatbot/chatbot.py';

    // 3. Komutu oluştur
    $command = "$python $script \"$input\"";

    $output = shell_exec($command . " 2>&1");
    if (empty($output)) {
    echo "<pre style='color:red'>[HATA] Python script çalışmadı veya boş çıktı verdi.</pre>";
} else {
    echo "<h3>Soru:</h3><p>$input</p>";
    echo "<h3>Cevap:</h3><pre>$output</pre>";
}

}

echo '
    <form method="get">
        <label>Soru Sor:</label><br>
        <input type="text" name="q" style="width: 50%;" required>
        <button type="submit">Gönder</button>
    </form>
';

echo $OUTPUT->footer();

