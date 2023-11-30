package com.ece420.english2morse;

import java.util.*;
public final class Constants {

    private Constants() {}
    public static final Map<Character, String> morseMap = new HashMap<Character, String>() {{
        put('A', ".-");
        put('B', "-...");
        put('C', "-.-.");
        put('D', "-..");
        put('E', ".");
        put('F', "..-.");
        put('G', "--.");
        put('H', "....");
        put('I', "..");
        put('J', ".---");
        put('K', "-.-");
        put('L', ".-..");
        put('M', "--");
        put('N', "-.");
        put('O', "---");
        put('P', ".--.");
        put('Q', "--.-");
        put('R', ".-.");
        put('S', "...");
        put('T', "-");
        put('U', "..-");
        put('V', "...-");
        put('W', ".--");
        put('X', "-..-");
        put('Y', "-.--");
        put('Z', "--..");
    }};

    public static final Map<Character, String> audioFileMap = new HashMap<Character, String>() {{
        put('.', "dot.wav");
        put('-', "dash.wav");
    }};
}


