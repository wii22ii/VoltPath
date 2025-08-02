//VoltPath Team's Arduino Prototype Code
 
void setup() {
  Serial.begin(9600);
  for (byte a = 2; a <= 6; a++) {
    pinMode(a, OUTPUT);
  }
}
void loop() {
  int value = analogRead(A1);
  Serial.println(value);
 
  for (int a = 1; a <= 5; a++) {
    if (value > a * 20) {
      digitalWrite(a + 1, HIGH);
    } else {
      digitalWrite(a + 1, LOW);
    }
  }
}
