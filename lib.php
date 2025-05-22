<?php

defined('MOODLE_INTERNAL') || die();

function local_chatbot_extend_navigation(global_navigation $nav) {
    $url = new moodle_url('/local/chatbot/index.php');
    $node = navigation_node::create(
        '🧠 AI Destekli Chatbot',
        $url,
        navigation_node::TYPE_SETTING,
        null,
        'local_chatbot'
    );
    $nav->add_node($node);
}





