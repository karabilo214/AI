// 9.36: действия помощника на Mac — окна, клик, перетаскивание, текст, клавиши, прокрутка.
// Сборка: swiftc -O act.swift -o act
//   act window <приложение>            — id x y w h самого большого обычного окна
//   act move <x> <y>                   — курсор (клавиши часть программ отдают области под мышью)
//   act click <x> <y> [double]         — координаты экрана в точках
//   act drag <x1> <y1> <x2> <y2>
//   act type <текст>
//   act key <cmd+shift+s | return | escape | left ...>
//   act scroll <x> <y> <dy>
import AppKit
import CoreGraphics
import Foundation

let args = CommandLine.arguments
func post(_ e: CGEvent?) { e?.post(tap: .cghidEventTap); usleep(30000) }
func mouse(_ type: CGEventType, _ p: CGPoint, clicks: Int64 = 1) {
  let e = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: p, mouseButton: .left)
  e?.setIntegerValueField(.mouseEventClickState, value: clicks)
  post(e)
}
func num(_ i: Int) -> Double { Double(args[i])! }

let keys: [String: CGKeyCode] = [
  "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
  "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
  "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40,
  "n": 45, "m": 46, "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
  "escape": 53, "esc": 53, "left": 123, "right": 124, "down": 125, "up": 126, "home": 115, "end": 119,
  "pageup": 116, "pagedown": 121, "=": 24, "-": 27, ".": 47, ",": 43, "/": 44, "f1": 122, "f2": 120,
]

switch args[1] {
case "window":
  let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as! [[String: Any]]
  var best: (Int, CGRect)? = nil
  for w in list where (w["kCGWindowOwnerName"] as? String) == args[2] && (w["kCGWindowLayer"] as? Int) == 0 {
    let b = w["kCGWindowBounds"] as! [String: CGFloat]
    let r = CGRect(x: b["X"]!, y: b["Y"]!, width: b["Width"]!, height: b["Height"]!)
    if best == nil || r.width * r.height > best!.1.width * best!.1.height { best = (w["kCGWindowNumber"] as! Int, r) }
  }
  if let (id, r) = best { print(id, Int(r.minX), Int(r.minY), Int(r.width), Int(r.height)) } else { exit(1) }
case "move":
  mouse(.mouseMoved, CGPoint(x: num(2), y: num(3)))
case "click":
  let p = CGPoint(x: num(2), y: num(3))
  mouse(.mouseMoved, p)
  let n: Int64 = args.count > 4 && args[4] == "double" ? 2 : 1
  for c in 1...n { mouse(.leftMouseDown, p, clicks: c); mouse(.leftMouseUp, p, clicks: c) }
case "drag":
  let a = CGPoint(x: num(2), y: num(3)), b = CGPoint(x: num(4), y: num(5))
  mouse(.mouseMoved, a); mouse(.leftMouseDown, a)
  for i in 1...20 {
    let t = Double(i) / 20
    mouse(.leftMouseDragged, CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t))
  }
  mouse(.leftMouseUp, b)
case "type":
  // Через буфер обмена и ⌘V: посимвольные события принимают не все программы (TextEdit — нет).
  let board = NSPasteboard.general
  let saved = board.string(forType: .string)
  board.clearContents()
  board.setString(args[2], forType: .string)
  usleep(80000)
  for down in [true, false] {
    let e = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: down)  // v
    e?.flags = .maskCommand
    post(e)
  }
  usleep(250000)
  board.clearContents()
  if let s = saved { board.setString(s, forType: .string) }
case "key":
  var flags: CGEventFlags = []
  var code: CGKeyCode? = nil
  for part in args[2].lowercased().split(separator: "+").map(String.init) {
    switch part {
    case "cmd", "command": flags.insert(.maskCommand)
    case "shift": flags.insert(.maskShift)
    case "alt", "option": flags.insert(.maskAlternate)
    case "ctrl", "control": flags.insert(.maskControl)
    default: code = keys[part]
    }
  }
  // Несколько клавиш подряд без модификаторов («0.3», «gx»): каждая буква — своя клавиша.
  var codes: [CGKeyCode] = []
  if let k = code {
    codes = [k]
  } else if flags.isEmpty, args[2].count > 1 {
    for ch in args[2].lowercased() {
      guard let k = keys[String(ch)] else { print("unknown key"); exit(2) }
      codes.append(k)
    }
  } else {
    print("unknown key"); exit(2)
  }
  // Символ клавиши задаётся явно: при русской раскладке код «g» дал бы «п», и программы,
  // которые смотрят на символ (Blender), не узнают сочетание.
  let chars: [CGKeyCode: String] = Dictionary(keys.filter { $0.key.count == 1 }.map { ($0.value, $0.key) },
                                              uniquingKeysWith: { a, _ in a })
  for k in codes {
    for down in [true, false] {
      let e = CGEvent(keyboardEventSource: nil, virtualKey: k, keyDown: down)
      e?.flags = flags
      if let ch = chars[k] {
        var units = Array(ch.utf16)
        e?.keyboardSetUnicodeString(stringLength: units.count, unicodeString: &units)
      }
      post(e)
    }
  }
case "scroll":
  mouse(.mouseMoved, CGPoint(x: num(2), y: num(3)))
  post(CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: Int32(num(4)), wheel2: 0, wheel3: 0))
default:
  exit(2)
}
